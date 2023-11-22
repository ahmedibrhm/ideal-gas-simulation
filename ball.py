import numpy as np
class Ball():
    def __init__(self, position, velocity, mass, radius=0.1):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
    
    def calculate_Kinetic_energy(self):
        "calculates kinetic energy"
        velocity_squared = sum([component ** 2 for component in self.velocity])
        return 0.5 * self.mass * velocity_squared
    
    def get_distance(self, position):
        squared_distance = 0
        for i in range(len(position)):
            x_1 = position[i]
            x_2 = self.position[i]
            d = (x_1-x_2)**2
            squared_distance += d
        return squared_distance**0.5

    def get_new_position(self, t):
        """move to a new position after time t"""
        new_position = [self.position[i] + self.velocity[i] * t for i in range(len(self.position))]
        self.position = new_position
        return self.position

    def border_collision(self, size):
        momentum_change = 0
        for i in range(len(self.position)):
            # Check for collision with either wall in this dimension
            if self.position[i] - self.radius <= 0 or self.position[i] + self.radius >= size:
                # Calculate the change in momentum
                momentum_change += abs(2 * self.mass * self.velocity[i])
                # Reverse the velocity in this dimension
                self.velocity[i] *= -1
        return momentum_change


    def is_collision(self, other_ball):
        # Calculate squared distance between balls
        squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(self.position, other_ball.position))
        # Calculate squared sum of radii
        radii_sum_squared = (self.radius + other_ball.radius) ** 2
        # Check for collision
        return squared_distance < radii_sum_squared

    def collide(self, other_ball):
        # Calculate the normal unit vector
        normal = np.array(other_ball.position) - np.array(self.position)
        normal /= np.linalg.norm(normal)

        # Relative velocity
        relative_velocity = np.array(other_ball.velocity) - np.array(self.velocity)

        # Velocity along the normal
        vel_along_normal = np.dot(relative_velocity, normal)

        # Do not resolve if velocities are separating
        if vel_along_normal > 0:
            return

        # Calculate impulse scalar (assuming perfectly elastic collision)
        impulse_scalar = -2 * vel_along_normal / (self.mass + other_ball.mass)

        # Apply impulse to each ball's velocity along the normal
        self.velocity -= (impulse_scalar * other_ball.mass * normal).tolist()
        other_ball.velocity += (impulse_scalar * self.mass * normal).tolist()
    
    def update_radius(self, radius):
        self.radius = radius
    
    def update_mass(self, mass):
        self.mass = mass
    
    def update_velocity(self, velocity):
        self.velocity = velocity
    
    def update_position(self, position):
        self.position = position