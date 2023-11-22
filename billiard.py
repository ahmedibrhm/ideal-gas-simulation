from ball import Ball
import random
from math import sqrt
import numpy as np

class BilliardModel():
    def __init__(self, number_of_particles=200, dimensions=2, size=1.0, temperature=300, delta_t=0.001, radius=0.1, mass = 1.0e-22):
        self.size = size
        self.k_B = 1.38064852*10**-23  # Boltzmann constant
        self.temperature = temperature
        self.dimensions = dimensions
        self.N = number_of_particles
        self.delta_t = delta_t
        self.mass = mass
        self.radius = radius
        self.energy = self.convert_temperature_to_energy()
        self.balls = self.create_balls()
        self.total_momentum_change = 0

    def create_balls(self):
        balls = []
        velocities = self.generate_velocities()
        positions = self.generate_positions()
        for i in range(self.N):
            position = positions[i]
            velocity = velocities[i]
            ball = Ball(position=position, velocity=velocity, mass=self.mass, radius=self.radius)
            balls.append(ball)
        return balls

    def generate_positions(self):
        """generate random N positions"""
        
        positions = []
        for i in range(self.N):
            position = [random.uniform(0, self.size) for _ in range(self.dimensions)]
            positions.append(position)
        return positions
    
    def generate_velocities(self):
        velocities = []
        total_kinetic_energy = 0
        # Generate random velocities according to normal distribution
        for _ in range(self.N):
            velocity = np.array([random.uniform(-1, 1) for i in range(self.dimensions)])
            velocities.append(velocity)
            total_kinetic_energy += 0.5 * self.mass * np.sum(velocity**2)
        energy_scaling_factor = sqrt(self.energy / total_kinetic_energy)
        for i in range(self.N):
            velocities[i] *= energy_scaling_factor
        return velocities

    def start_simulation(self, end_time=1000, skip_rate=50):
        # Calculate the total number of frames based on end_time and delta_t
        total_frames = int(end_time / self.delta_t)

        # Define a desired frame interval in terms of simulation time
        desired_frame_interval = 10 # You can adjust this value

        # Calculate the skip rate based on the desired frame interval and delta_t
        skip_rate = int(desired_frame_interval / self.delta_t)

        frames = []
        for i in range(total_frames):
            if i % skip_rate == 0:
                frame_data = {
                    "positions": [[ball.position[j] for j in range(len(ball.position))] for ball in self.balls],
                    "temperature": self.get_temperature(),
                    "pressure": self.get_total_pressure(),
                    "total_energy": self.get_total_energy()
                }
                frames.append(frame_data)
            self.next_frame()
        return frames

    def next_frame(self):
        # Updated method to include particle-particle collision
        self.handle_particle_collisions()
        total_momentum_change = 0
        for ball in self.balls:
            momentum_change = ball.border_collision(self.size)
            total_momentum_change += momentum_change
            ball.get_new_position(self.delta_t)
        self.total_momentum_change = total_momentum_change

    def get_total_pressure(self):
        dimensions = self.dimensions
        surface_area = 2**dimensions * self.size**(dimensions - 1)
        pressure = self.total_momentum_change / (surface_area * self.delta_t)
        return pressure
    
    def handle_particle_collisions(self):
        for i, ball_1 in enumerate(self.balls):
            for ball_2 in self.balls[i+1:]:
                if ball_1.is_collision(ball_2):
                    ball_1.collide(ball_2)

    def get_total_energy(self):
        """calculate the total energy of the system"""
        total_energy = sum([ball.calculate_Kinetic_energy() for ball in self.balls])
        return total_energy

    def get_temperature(self):
        """calculate the temperature as an analogy for average kinetic energy"""
        total_energy = self.get_total_energy()
        average_energy_per_particle = total_energy / self.N
        dof = self.dimensions  # degrees of freedom, changes with dimensions
        temperature = (2/dof) * average_energy_per_particle / self.k_B
        return temperature

    def get_shape(self):
        """calculate the volume of the container in any dimensions"""
        return self.size ** self.dimensions

    def real_time_simulation(self):
        while True:
            frame_data = {
                "positions": [[ball.position[j] for j in range(len(ball.position))] for ball in self.balls],
                "temperature": self.get_temperature(),
                "pressure": self.get_total_pressure(),
                "total_energy": self.get_total_energy()
            }
            yield frame_data
            self.next_frame()
    
    def convert_temperature_to_energy(self):
        """get the energy from the tempreture"""
        energy = (self.dimensions / 2) * self.temperature * self.k_B * self.N
        return energy

    def update_delta_t(self, new_delta_t):
        self.delta_t = new_delta_t

    def update_balls(self, velocities=False):
        """update if the arg isn't None"""
        if velocities:
            velocities = self.generate_velocities()
            
        for i in range(len(self.balls)):
            ball = self.balls[i]
            if velocities:
                ball.update_velocity(velocities[i])
            ball.update_mass(self.mass)
            ball.update_radius(self.radius)


    def update_mass(self, new_mass):
        self.mass = new_mass
        self.update_balls()

    def update_radius(self, new_radius):
        self.radius = new_radius
        self.update_balls()

    def update_temperature(self, new_temperature):
        self.temperature = new_temperature
        self.energy = self.convert_temperature_to_energy()
        self.update_balls(velocities=True)