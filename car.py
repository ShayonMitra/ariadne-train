import numpy as np

class DifferentialDriveRobot:
    def __init__(self, wheel_radius, wheelbase, dt):
        self.R = wheel_radius  # Wheel radius
        self.L = wheelbase     # Wheelbase
        self.dt = dt           # Time step
        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]

    def reset(self, x, y, theta):
        """Reset the robot's state to the specified position and orientation."""
        self.state = np.array([x, y, theta])

    def step(self, omega_L, omega_R):
        """Simulate one time step of the robot's motion."""
        x, y, theta = self.state
        v = self.R / 2 * (omega_R + omega_L)  # Linear velocity
        omega = self.R / self.L * (omega_R - omega_L)  # Angular velocity

        # Update position and orientation
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += omega * self.dt

        # Wrap theta to [-pi, pi]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        self.state = np.array([x, y, theta])
        return self.state

    def get_state(self):
        """Return the current state of the robot."""
        return self.state


def simulate_to_goal(robot, start, goal, tolerance=0.1, max_steps=1000):
    """
    Simulate the robot's motion from start to goal.
    :param robot: Instance of DifferentialDriveRobot.
    :param start: 2D start position as [x, y].
    :param goal: 2D goal position as [x, y].
    :param tolerance: Distance tolerance for reaching the goal.
    :param max_steps: Maximum number of simulation steps.
    :return: Time taken to reach the goal or None if not reached within max_steps.
    """
    elapsed_time = 0.0
    robot.reset(x=start[0], y=start[1], theta=0.0)  # Reset robot to start position

    start = np.array(start)
    goal = np.array(goal)
    for step in range(max_steps):
        # Get current state
        x, y, theta = robot.get_state()

        # Calculate distance and angle to goal
        goal_distance = np.linalg.norm([goal[0] - x, goal[1] - y])
        goal_angle = np.arctan2(goal[1] - y, goal[0] - x)

        # Check if within tolerance
        if goal_distance <= tolerance:
            return elapsed_time

        # Proportional control for angular and linear velocity
        k_linear = 1.0  # Gain for linear velocity
        k_angular = 2.0  # Gain for angular velocity

        angle_error = goal_angle - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Wrap to [-pi, pi]

        linear_velocity = k_linear * goal_distance
        angular_velocity = k_angular * angle_error

        # Convert to wheel velocities
        omega_L = (linear_velocity - (robot.L / 2) * angular_velocity) / robot.R
        omega_R = (linear_velocity + (robot.L / 2) * angular_velocity) / robot.R

        # Step the robot
        robot.step(omega_L, omega_R)

        # Increment elapsed time
        elapsed_time += robot.dt

    print("Goal not reached within max steps!")
    return None


# Example usage
if __name__ == "__main__":
    # Create the robot
    robot = DifferentialDriveRobot(wheel_radius=0.05, wheelbase=0.15, dt=0.1)

    # Define the start and goal positions
    start_position = [10.0, 15.0]  # Starting at the origin
    goal_position = [10.0, 10.0]  # Goal at (10.0, 10.0)

    # Simulate to goal
    time_taken = simulate_to_goal(robot, start=start_position, goal=goal_position)
    if time_taken is not None:
        print(f"Time taken to reach the goal: {time_taken:.2f} seconds")
    else:
        print("Failed to reach the goal.")
