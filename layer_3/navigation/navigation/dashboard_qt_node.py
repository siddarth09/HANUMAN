#!/usr/bin/env python3
"""HANUMAN Mars navigation console — PyQt5, custom-painted over a full-bleed map."""
import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import (QPainter, QColor, QPen, QFont, QFontMetrics, QImage,
                         QPolygonF, QRadialGradient, QLinearGradient)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF

ALBEDO = ("/home/sid/projects25/src/HANUMAN/mars_gazebo/unitree_g1_mjcf/"
          "mars_nav_200/mars_nav_200_albedo.png")
DEM_CACHE = "/tmp/hanuman_dem.npz"
ALB_EXTENT = (-100.0, -28.0, 100.0, 172.0)   # albedo coverage in map frame (x0,y0,x1,y1)

C = {
    "bg": "#070503", "terrain": "#241510", "terrain_edge": "#0a0604",
    "line": "#3a2a1d", "speckle": "#c97a3a",
    "white": "#ffffff", "dim": "#8a6f5a",
    "accent": "#ff7b00", "accent_lt": "#ffae57",
    "gtsam": "#e8d5c5", "ekf": "#7fbfff", "truth": "#ffffff",
    "goal": "#ff5db1", "path": "#5ce1a6", "mppi": "#ffae57",
    "ok": "#5ce1a6", "warn": "#ffae57", "crit": "#ff4d4d",
}


def qc(name, a=255):
    c = QColor(C.get(name, name))
    c.setAlpha(a)
    return c


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class DashboardNode(Node):
    def __init__(self):
        super().__init__("hanuman_dashboard")
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.msg = {}
        self.stamp = {}
        self._sub(OccupancyGrid, "/nav/global_costmap", latched)
        self._sub(Path, "/nav/global_path", 1)
        self._sub(Path, "/nav/mppi_path", 1)
        self._sub(PoseWithCovarianceStamped, "/terrain_match/pose", 10)
        self._sub(Odometry, "/odometry/filtered", 10)
        self._sub(Odometry, "/ground_truth/odom", 10)
        self._sub(Twist, "/cmd_vel", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 1)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)

    def _sub(self, typ, topic, qos):
        self.create_subscription(typ, topic, lambda m, t=topic: self._on(t, m), qos)

    def _on(self, topic, m):
        self.msg[topic] = m
        self.stamp[topic] = self.now()

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def link_ok(self, within=1.0):
        return any(self.now() - t < within for t in self.stamp.values()) if self.stamp else False

    def pose_xy(self, topic, cov=False):
        m = self.msg.get(topic)
        if m is None:
            return None
        p = m.pose.pose
        out = (p.position.x, p.position.y, yaw_of(p.orientation))
        if cov:
            c = m.pose.covariance[0]
            return out, math.sqrt(c) if c > 0 else 0.0
        return out

    def publish_goal(self, x, y, yaw):
        m = PoseStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.orientation.z = math.sin(yaw / 2)
        m.pose.orientation.w = math.cos(yaw / 2)
        self.goal_pub.publish(m)

    def publish_initialpose(self, x, y, yaw):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
        m.pose.pose.orientation.z = math.sin(yaw / 2)
        m.pose.pose.orientation.w = math.cos(yaw / 2)
        self.pose_pub.publish(m)


class MapView(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("HANUMAN — Mars Navigation")
        self.setMinimumSize(1100, 720)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{C['bg']};")

        try:
            d = np.load(DEM_CACHE)
            gx, gy = d["gx"], d["gy"]
            mgn = 2.0
            self.x0, self.x1 = float(gx[0]) - mgn, float(gx[-1]) + mgn
            self.y0, self.y1 = float(gy[0]) - mgn, float(gy[-1]) + mgn
        except Exception:
            self.x0, self.y0, self.x1, self.y1 = ALB_EXTENT

        self.albedo = QImage(ALBEDO) if os.path.exists(ALBEDO) else QImage()
        self._cost_img = None
        self._cost_buf = None
        self._cost_world = None
        self._cost_src = None

        self.tool = None            # 'goal' | 'pose' | None
        self.show_cost = True
        self.goal = None            # (x, y, yaw)
        self._press = None          # press pixel
        self._cursor = None         # current pixel (drag preview)
        self._btns = {}

        rng = np.random.default_rng(7)
        self._speckle = rng.random((600, 2))

    # ---- transform ----
    def _fit(self):
        W, H = self.width(), self.height()
        s = min(W / (self.x1 - self.x0), H / (self.y1 - self.y0))
        ox = (W - s * (self.x1 - self.x0)) / 2
        oy = (H - s * (self.y1 - self.y0)) / 2
        return s, ox, oy

    def toPx(self, mx, my):
        s, ox, oy = self._fit()
        return (ox + (mx - self.x0) * s, self.height() - oy - (my - self.y0) * s)

    def toWorld(self, px, py):
        s, ox, oy = self._fit()
        return (self.x0 + (px - ox) / s, self.y0 + (self.height() - py - oy) / s)

    # ---- fonts / primitives ----
    def _font(self, size, weight=QFont.Light, track=0.0):
        f = QFont("Inter")
        f.setPixelSize(size)
        f.setWeight(weight)
        if track:
            f.setLetterSpacing(QFont.AbsoluteSpacing, track)
        return f

    def _txt(self, p, x, y, s, color, size, weight=QFont.Light, track=0.0, right=False):
        f = self._font(size, weight, track)
        p.setFont(f)
        p.setPen(qc(color))
        if right:
            x -= QFontMetrics(f).horizontalAdvance(s)
        p.drawText(int(x), int(y), s)

    def _rule(self, p, x, y, w):
        p.setPen(QPen(qc("line"), 1))
        p.drawLine(int(x), int(y), int(x + w), int(y))

    def _bar(self, p, x, y, w, h, frac, color):
        p.setPen(QPen(qc("line"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(int(x), int(y), int(w), int(h))
        fw = max(0.0, min(1.0, frac)) * w
        p.fillRect(QRectF(x, y, fw, h), qc(color))

    def _swatch(self, p, x, y, color, hollow=False):
        if hollow:
            p.setPen(QPen(qc(color), 1.5))
            p.setBrush(Qt.NoBrush)
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(qc(color))
        p.drawRect(int(x), int(y) - 10, 10, 10)

    def _star(self, cx, cy, r):
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append(QPointF(cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        return QPolygonF(pts)

    # ---- costmap pre-render (once per latched grid) ----
    def _render_cost(self, g):
        W, H = g.info.width, g.info.height
        data = np.array(g.data, dtype=np.int16).reshape(H, W)
        c = np.clip(data, 0, 100).astype(np.float32) / 100.0
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = (123 * (1 - c) + 77 * c).astype(np.uint8)     # amber -> red
        rgba[..., 2] = (77 * c).astype(np.uint8)
        rgba[..., 3] = (c * 190).astype(np.uint8)
        rgba[data < 0] = 0
        rgba = np.ascontiguousarray(np.flipud(rgba))                 # row 0 -> max y
        self._cost_buf = rgba
        self._cost_img = QImage(rgba.data, W, H, 4 * W, QImage.Format_RGBA8888)
        ox, oy = g.info.origin.position.x, g.info.origin.position.y
        r = g.info.resolution
        self._cost_world = (ox, oy, ox + W * r, oy + H * r)
        self._cost_src = g

    def _img_rect(self, x0, y0, x1, y1):
        tl = self.toPx(x0, y1)
        br = self.toPx(x1, y0)
        return QRectF(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])

    # ---- paint ----
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        W, H = self.width(), self.height()

        self._paint_terrain(p, W, H)
        cm = self.node.msg.get("/nav/global_costmap")
        if cm is not None and cm is not self._cost_src:
            self._render_cost(cm)
        if self.show_cost and self._cost_img is not None:
            p.drawImage(self._img_rect(*self._cost_world), self._cost_img)
        self._paint_grid(p)
        self._paint_paths(p)
        self._paint_goal(p)
        self._paint_poses(p)
        self._paint_drag(p)

        self._paint_topbar(p, W)
        self._paint_localization(p)
        self._paint_navigation(p, W)
        self._paint_legend(p, H)
        self._paint_tools(p, W, H)
        p.end()

    def _paint_terrain(self, p, W, H):
        if not self.albedo.isNull():
            p.fillRect(self.rect(), qc("bg"))
            p.drawImage(self._img_rect(*ALB_EXTENT), self.albedo)
            return
        g = QRadialGradient(W / 2, H / 2, max(W, H) * 0.7)
        g.setColorAt(0, qc("terrain"))
        g.setColorAt(1, qc("terrain_edge"))
        p.fillRect(self.rect(), g)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("speckle", 22))
        for sx, sy in self._speckle:
            p.drawRect(int(sx * W), int(sy * H), 1, 1)

    def _paint_grid(self, p):
        p.setPen(QPen(qc("white", 10), 1))
        x = math.ceil(self.x0 / 20) * 20
        while x <= self.x1:
            a = self.toPx(x, self.y0)
            b = self.toPx(x, self.y1)
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            x += 20
        y = math.ceil(self.y0 / 20) * 20
        while y <= self.y1:
            a = self.toPx(self.x0, y)
            b = self.toPx(self.x1, y)
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            y += 20

    def _poly(self, path_msg):
        return QPolygonF([QPointF(*self.toPx(ps.pose.position.x, ps.pose.position.y))
                          for ps in path_msg.poses])

    def _paint_paths(self, p):
        gp = self.node.msg.get("/nav/global_path")
        if gp and gp.poses:
            p.setPen(QPen(qc("path"), 1.5))
            p.drawPolyline(self._poly(gp))
        mp = self.node.msg.get("/nav/mppi_path")
        if mp and mp.poses:
            pen = QPen(qc("mppi"), 2)
            pen.setDashPattern([4, 4])
            p.setPen(pen)
            p.drawPolyline(self._poly(mp))

    def _paint_goal(self, p):
        if self.goal is None:
            return
        cx, cy = self.toPx(self.goal[0], self.goal[1])
        p.setPen(QPen(qc("goal"), 3))
        p.drawLine(int(cx - 10), int(cy - 10), int(cx + 10), int(cy + 10))
        p.drawLine(int(cx - 10), int(cy + 10), int(cx + 10), int(cy - 10))

    def _heading_line(self, p, x, y, yaw, color, ln=26):
        p.setPen(QPen(qc(color), 1.5))
        p.drawLine(int(x), int(y), int(x + ln * math.cos(yaw)), int(y - ln * math.sin(yaw)))

    def _paint_poses(self, p):
        s, _, _ = self._fit()
        gt = self.node.pose_xy("/ground_truth/odom")
        if gt:
            x, y = self.toPx(gt[0], gt[1])
            p.setPen(QPen(qc("truth"), 1.8))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(x, y), 9, 9)
        ekf = self.node.pose_xy("/odometry/filtered")
        if ekf:
            x, y = self.toPx(ekf[0], ekf[1])
            p.setPen(Qt.NoPen)
            p.setBrush(qc("ekf"))
            p.drawEllipse(QPointF(x, y), 6.5, 6.5)
            self._heading_line(p, x, y, ekf[2], "ekf")
        g = self.node.pose_xy("/terrain_match/pose", cov=True)
        best = None
        if g:
            (gx, gy, gyaw), cov = g
            x, y = self.toPx(gx, gy)
            best = (x, y)
            if cov > 0:
                pen = QPen(qc("gtsam", 110), 1)
                pen.setDashPattern([3, 3])
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                rr = max(4, cov * s)
                p.drawEllipse(QPointF(x, y), rr, rr)
            p.setPen(QPen(qc("white"), 1.2))
            p.setBrush(qc("gtsam"))
            p.drawPolygon(self._star(x, y, 13))
            self._heading_line(p, x, y, gyaw, "gtsam", 28)
        elif ekf:
            best = self.toPx(ekf[0], ekf[1])
        if best:
            pen = QPen(qc("dim", 160), 1)
            pen.setDashPattern([2, 4])
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(best[0], best[1]), 0.45 * s, 0.45 * s)

    def _paint_drag(self, p):
        if self._press is None or self.tool is None or self._cursor is None:
            return
        color = "goal" if self.tool == "goal" else "ekf"
        x0, y0 = self._press
        p.setPen(QPen(qc(color), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(x0, y0), 6, 6)
        p.drawLine(int(x0), int(y0), int(self._cursor[0]), int(self._cursor[1]))

    # ---- HUD ----
    def _paint_topbar(self, p, W):
        g = QLinearGradient(0, 0, 0, 68)
        g.setColorAt(0, qc("bg", 235))
        g.setColorAt(1, qc("bg", 0))
        p.fillRect(QRectF(0, 0, W, 68), g)
        self._txt(p, 24, 44, "HANUMAN", "white", 28, QFont.Light, track=9)
        self._txt(p, 250, 42, "JEZERO CRATER · G1 SURFACE OPS", "dim", 13, QFont.Normal, track=3)
        ok = self.node.link_ok()
        p.setPen(Qt.NoPen)
        p.setBrush(qc("ok") if ok else qc("crit"))
        txt = "LINK NOMINAL · 10 HZ" if ok else "LINK LOST"
        f = self._font(13, QFont.Normal, 3)
        tw = QFontMetrics(f).horizontalAdvance(txt)
        p.drawRect(W - 28 - tw - 16, 30, 9, 9)
        self._txt(p, W - 24, 42, txt, "ok" if ok else "crit", 13, QFont.Normal, track=3, right=True)

    def _div_readout(self, p, x, y, label, est_color, err):
        self._swatch(p, x, y, est_color)
        self._txt(p, x + 18, y, label, "dim", 13, QFont.Normal, track=2)
        if err is None:
            self._txt(p, x + 330, y, "—", "dim", 26, right=True)
            return
        col = "white" if err < 0.3 else ("warn" if err < 0.8 else "crit")
        self._txt(p, x + 330, y, f"{err:.3f}", col, 26, QFont.Light, right=True)
        self._bar(p, x + 210, y + 8, 120, 4, err / 1.5, col)

    def _paint_localization(self, p):
        x, y = 24, 96
        self._txt(p, x, y, "LOCALIZATION", "accent", 14, QFont.Normal, track=4)
        self._rule(p, x, y + 12, 330)
        gt = self.node.pose_xy("/ground_truth/odom")
        g = self.node.pose_xy("/terrain_match/pose")
        ekf = self.node.pose_xy("/odometry/filtered")
        gerr = math.hypot(g[0] - gt[0], g[1] - gt[1]) if (g and gt) else None
        eerr = math.hypot(ekf[0] - gt[0], ekf[1] - gt[1]) if (ekf and gt) else None
        self._div_readout(p, x, y + 52, "GTSAM · VS TRUTH", "gtsam", gerr)
        self._div_readout(p, x, y + 92, "EKF · VS TRUTH", "ekf", eerr)
        self._txt(p, x, y + 132, "TERRAIN-FIX CONFIDENCE", "dim", 12, QFont.Normal, track=2)
        gc = self.node.pose_xy("/terrain_match/pose", cov=True)
        conf = max(0.0, 100 - (gc[1] * 80)) if gc else 0.0
        cc = "ok" if conf > 60 else ("warn" if conf > 30 else "crit")
        self._bar(p, x, y + 142, 320, 6, conf / 100.0, cc)
        self._txt(p, x + 330, y + 138, f"{conf:.0f}%", cc, 16, QFont.Light, right=True)
        if gerr is not None and gerr >= 0.8:
            p.setPen(QPen(qc("crit"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(x, y + 164, 340, 32)
            self._txt(p, x + 12, y + 185, "GTSAM DIVERGED — RE-ANCHOR POSE", "crit", 13,
                      QFont.Normal, track=1)

    def _paint_navigation(self, p, W):
        x, y = W - 334, 96
        self._txt(p, x, y, "NAVIGATION", "accent", 14, QFont.Normal, track=4)
        self._rule(p, x, y + 12, 310)
        g = self.node.pose_xy("/terrain_match/pose")
        dist = math.hypot(self.goal[0] - g[0], self.goal[1] - g[1]) if (self.goal and g) else None
        herr = None
        if self.goal and g:
            bearing = math.atan2(self.goal[1] - g[1], self.goal[0] - g[0])
            herr = math.degrees(math.atan2(math.sin(bearing - g[2]), math.cos(bearing - g[2])))
        self._txt(p, x, y + 48, "DIST TO GOAL", "dim", 12, QFont.Normal, track=2)
        self._txt(p, x + 310, y + 54, (f"{dist:.1f} M" if dist is not None else "—"),
                  "accent_lt", 32, QFont.Light, right=True)
        self._txt(p, x, y + 90, "HEADING ERR", "dim", 12, QFont.Normal, track=2)
        hc = "warn" if (herr is not None and abs(herr) > 20) else "white"
        self._txt(p, x + 310, y + 92, (f"{herr:+.0f}°" if herr is not None else "—"),
                  hc, 22, QFont.Light, right=True)
        cmd = self.node.msg.get("/cmd_vel")
        vx = cmd.linear.x if cmd else 0.0
        wz = cmd.angular.z if cmd else 0.0
        self._txt(p, x, y + 126, "LINEAR VX", "dim", 12, QFont.Normal, track=2)
        self._txt(p, x + 310, y + 126, f"{vx:+.2f}", "accent_lt", 15, QFont.Light, right=True)
        self._bar(p, x, y + 134, 310, 6, abs(vx) / 0.4, "accent_lt")
        self._txt(p, x, y + 164, "ANGULAR WZ", "dim", 12, QFont.Normal, track=2)
        self._txt(p, x + 310, y + 164, f"{wz:+.2f}", "accent_lt", 15, QFont.Light, right=True)
        self._bar(p, x, y + 172, 310, 6, abs(wz) / 0.8, "accent_lt")
        if self.goal is None:
            mode = "IDLE"
        elif dist is not None and dist < 0.5:
            mode = "GOAL REACHED"
        elif herr is not None and abs(herr) > 20:
            mode = "TURN-IN-PLACE"
        else:
            mode = "TRANSIT"
        self._txt(p, x, y + 206, "MODE", "dim", 12, QFont.Normal, track=2)
        self._txt(p, x + 310, y + 208, mode, "accent", 18, QFont.Normal, track=2, right=True)

    def _paint_legend(self, p, H):
        rows = [("GTSAM", "gtsam"), ("EKF", "ekf"), ("TRUTH", "truth"),
                ("GLOBAL PATH", "path"), ("MPPI", "mppi"), ("GOAL", "goal")]
        x, y = 24, H - 96 - len(rows) * 24
        for i, (lab, c) in enumerate(rows):
            yy = y + i * 24
            self._swatch(p, x, yy, c, hollow=(lab == "TRUTH"))
            self._txt(p, x + 18, yy, lab, "dim", 12, QFont.Normal, track=2)

    def _tool_rects(self, W, H):
        labels = ["SET GOAL", "RE-ANCHOR", "CLEAR", "COST"]
        bw, bh, gap = 156, 46, 14
        total = len(labels) * bw + (len(labels) - 1) * gap
        x = (W - total) / 2
        y = H - 76
        rects = {}
        for lab in labels:
            rects[lab] = QRectF(x, y, bw, bh)
            x += bw + gap
        return rects

    def _paint_tools(self, p, W, H):
        self._btns = self._tool_rects(W, H)
        active = {"SET GOAL": (self.tool == "goal", "goal"),
                  "RE-ANCHOR": (self.tool == "pose", "ekf"),
                  "CLEAR": (False, "dim"),
                  "COST": (self.show_cost, "accent")}
        for lab, r in self._btns.items():
            on, color = active[lab]
            if on:
                p.fillRect(r, qc(color))
                tcol = "bg"
            else:
                p.setPen(QPen(qc("line"), 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(r)
                tcol = "white"
            f = self._font(13, QFont.Normal, 3)
            p.setFont(f)
            p.setPen(qc(tcol))
            tw = QFontMetrics(f).horizontalAdvance(lab)
            p.drawText(int(r.center().x() - tw / 2), int(r.center().y() + 5), lab)
        hint = {"goal": "CLICK-DRAG TO SET DESTINATION + HEADING",
                "pose": "CLICK-DRAG TO RE-ANCHOR LOCALIZATION (2D POSE)"}.get(self.tool, "")
        if hint:
            f = self._font(11, QFont.Normal, 2)
            p.setFont(f)
            p.setPen(qc("dim"))
            tw = QFontMetrics(f).horizontalAdvance(hint)
            p.drawText(int(W / 2 - tw / 2), H - 20, hint)

    # ---- interaction ----
    def mousePressEvent(self, e):
        pt = e.pos()
        for lab, r in self._btns.items():
            if r.contains(QPointF(pt)):
                self._on_button(lab)
                self.update()
                return
        if self.tool:
            self._press = (pt.x(), pt.y())
            self._cursor = (pt.x(), pt.y())

    def mouseMoveEvent(self, e):
        self._cursor = (e.pos().x(), e.pos().y())
        if self._press:
            self.update()

    def mouseReleaseEvent(self, e):
        if self._press is None or self.tool is None:
            self._press = None
            return
        x0, y0 = self._press
        dx, dy = e.pos().x() - x0, e.pos().y() - y0
        yaw = -math.atan2(dy, dx) if math.hypot(dx, dy) > 4 else 0.0
        wx, wy = self.toWorld(x0, y0)
        if self.tool == "goal":
            self.node.publish_goal(wx, wy, yaw)
            self.goal = (wx, wy, yaw)
        else:
            self.node.publish_initialpose(wx, wy, yaw)
        self._press = None
        self.tool = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _on_button(self, lab):
        if lab == "SET GOAL":
            self.tool = None if self.tool == "goal" else "goal"
        elif lab == "RE-ANCHOR":
            self.tool = None if self.tool == "pose" else "pose"
        elif lab == "CLEAR":
            self.goal = None
            self.tool = None
        elif lab == "COST":
            self.show_cost = not self.show_cost
        self.setCursor(Qt.CrossCursor if self.tool else Qt.ArrowCursor)


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    app = QApplication(sys.argv)
    view = MapView(node)
    view.showMaximized()

    timer = QTimer()
    timer.timeout.connect(lambda: (rclpy.spin_once(node, timeout_sec=0), view.update()))
    timer.start(100)

    try:
        app.exec_()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
