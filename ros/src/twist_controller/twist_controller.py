import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # Minimum throttle value
        mx = 0.2 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        kp_steer = 0.3
        ki_steer = 0.0
        kd_steer = 0.0
        mn_steer = -max_steer_angle/2 # Minimum steer value
        mx_steer = max_steer_angle/2 # Maximum steer value
        self.steer_controller = PID(kp_steer, ki_steer, kd_steer, mn_steer, mx_steer)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = .02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()


    def control(self, curr_lin_vel, linear_vel, curr_ang_vel ,angular_vel, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            self.steer_controller.reset()
            return 0., 0., 0.

        curr_lin_vel = self.vel_lpf.filt(curr_lin_vel)


        # rospy.logwarn("Angular velocity: {0}".format(angular_vel))
        # rospy.logwarn("Target velocity: {0}".format(linear_vel))
        # rospy.logwarn("Current velocity: {0}".format(current_vel))
        # rospy.logwarn("Filtered velocity: {0}".format(self.vel_lpf.get()))

        #steering = self.yaw_controller.get_steering(linear_vel, angular_vel, curr_lin_vel)

        vel_error = linear_vel - curr_lin_vel
        self.last_vel = curr_lin_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        
        ang_vel_error = angular_vel - curr_ang_vel
        steering = self.steer_controller.step(ang_vel_error, sample_time)
        
        brake = 0

        if linear_vel == 0 and curr_lin_vel < 0.1:
            throttle = 0
            brake = 700 #N*m - to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N*m

        return throttle, brake, steering
