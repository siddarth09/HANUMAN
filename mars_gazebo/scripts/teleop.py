#!/usr/bin/env python3
"""
HANUMAN Teleop — keyboard control for humanoid locomotion policy.

Publishes geometry_msgs/Twist to /cmd_vel.
The RL policy node subscribes and translates to walking gaits.

Controls:
    w/s : forward/backward
    a/d : strafe left/right
    q/e : turn left/right
    x   : stop (zero velocity)
    +/- : increase/decrease speed
    ESC : quit
"""
import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

HELP_TEXT = """
╔══════════════════════════════════════╗
║     HANUMAN Mars Teleop              ║
╠══════════════════════════════════════╣
║                                      ║
║          w : forward                 ║
║    a : left   d : right              ║
║          s : backward                ║
║                                      ║
║    q : turn left   e : turn right    ║
║    x : STOP                          ║
║                                      ║
║    +/= : speed up                    ║
║    -   : slow down                   ║
║                                      ║
║    ESC/Ctrl+C : quit                 ║
╚══════════════════════════════════════╝
"""

KEY_BINDINGS = {
    'w': ( 1.0,  0.0,  0.0),  # forward
    's': (-1.0,  0.0,  0.0),  # backward
    'a': ( 0.0,  1.0,  0.0),  # strafe left
    'd': ( 0.0, -1.0,  0.0),  # strafe right
    'q': ( 0.0,  0.0,  1.0),  # turn left
    'e': ( 0.0,  0.0, -1.0),  # turn right
    'x': ( 0.0,  0.0,  0.0),  # stop
}


class HanumanTeleop(Node):
    def __init__(self):
        super().__init__('hanuman_teleop')

        self.declare_parameter('max_linear', 0.5)    # m/s — conservative for humanoid
        self.declare_parameter('max_angular', 0.5)    # rad/s
        self.declare_parameter('speed_step', 0.1)
        self.declare_parameter('rate', 20.0)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.speed = 0.3   # current speed multiplier
        self.max_lin = self.get_parameter('max_linear').value
        self.max_ang = self.get_parameter('max_angular').value
        self.step = self.get_parameter('speed_step').value

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

    def get_key(self):
        """Read a single keypress from stdin."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def run(self):
        print(HELP_TEXT)
        print(f'  Speed: {self.speed:.1f} m/s  (max linear: {self.max_lin}, max angular: {self.max_ang})')
        print()

        try:
            while rclpy.ok():
                key = self.get_key()

                if key == '\x1b' or key == '\x03':  # ESC or Ctrl+C
                    # Send stop before quitting
                    self.publish_twist(0.0, 0.0, 0.0)
                    break

                if key in KEY_BINDINGS:
                    dx, dy, dw = KEY_BINDINGS[key]
                    self.vx = dx * self.speed
                    self.vy = dy * self.speed
                    self.wz = dw * self.max_ang

                elif key in ('+', '='):
                    self.speed = min(self.speed + self.step, self.max_lin)
                    print(f'\r  Speed: {self.speed:.1f} m/s   ', end='', flush=True)

                elif key == '-':
                    self.speed = max(self.speed - self.step, 0.0)
                    print(f'\r  Speed: {self.speed:.1f} m/s   ', end='', flush=True)

                else:
                    continue

                self.publish_twist(self.vx, self.vy, self.wz)

                if key == 'x':
                    print('\r  [STOP]                    ', end='', flush=True)
                elif key in KEY_BINDINGS:
                    print(f'\r  vx={self.vx:+.2f}  vy={self.vy:+.2f}  wz={self.wz:+.2f}  speed={self.speed:.1f}   ',
                          end='', flush=True)

        except Exception as e:
            self.get_logger().error(f'Teleop error: {e}')
        finally:
            # Always send stop on exit
            self.publish_twist(0.0, 0.0, 0.0)
            print('\nStopped.')

    def publish_twist(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = HanumanTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()