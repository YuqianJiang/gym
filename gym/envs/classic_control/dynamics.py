#! /usr/local/bin/python3
import gym
import numpy as np
import scipy.linalg
import time

env = gym.make('CartPole-v1').env
env.theta_threshold_radians = 90 * np.pi / 180

def linearized_dynamics():
    # Dynamics linearized at x: 0, xdot: 0, theta: 0, thetadot: 0
    # (This is the position of the pole at the top)

    # Using notation (but not equations, see why below) from https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf

    mc = env.masscart
    mp = env.masspole
    l = env.length
    g = env.gravity

    # It turns out that Sutton's cartpole is actually wrong lololol (see https://coneural.org/florian/papers/05_cart_pole.pdf)
    # So to be safe, let's just have Matlab derive the stuff from the dynamics in cartpole.py

    # A and B are given from the following Matlab script (run it if you don't believe me)
    # (n.b. You probably shouldn't believe me)
    '''
    syms mc mp g l;
    syms x xd t td;
    syms u;

    % From gym's cartpole.py
    tdotdot = (g * sin(t) - cos(t) * ((u + mp*l * (td^2) * sin(t)) / (mp + mc))) / (l * (4/3 - mp * cos(t)^2 / (mp + mc)));
    xdotdot = ((u + mp*l * td^2 * sin(t)) / (mp + mc)) - mp*l * tdotdot * cos(t) / (mp + mc);

    statedot = [xd; xdotdot; td; tdotdot];

    A = jacobian(statedot, [x xd t td]);
    B = jacobian(statedot, u);

    % We want to linearize at the unstable top position of the pole
    % i.e. [0 0 0 0]
    disp('A: ');
    disp(subs(A, [x xd t td], [0 0 0 0]));

    disp('B: ');
    disp(subs(B, [x xd t td], [0 0 0 0]));
    '''

    A = np.array([
        [ 0, 1,                                       0, 0],
        [ 0, 0, (g*mp)/((mc + mp)*(mp/(mc + mp) - 4.0/3.0)), 0],
        [ 0, 0,                                       0, 1],
        [ 0, 0,             -g/(l*(mp/(mc + mp) - 4.0/3.0)), 0]
    ])

    B = np.array([
       [0],
       [1/(mc + mp) - mp/(((mc + mp)**2)*(mp/(mc + mp) - 4/3))],
       [0],
       [1/(l*(mc + mp)*(mp/(mc + mp) - 4/3))]
    ])

    return A, B


def lqr(A, B, Q, R):
    # We're using notation from the LQR Wikipedia page
    # https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator (Section Infinite-Horizon Continuous LQR)
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    # Note that N in the Wiki article is zero here
    F = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    return F

# Some hyperparameters
# entries in Q penalize relative errors in state
# entries in R penalize actions
Q = np.diag([0.01, 5.0, 30.0, 2.0])
R = np.array([[0.5]])
A, B = linearized_dynamics()
K = lqr(A, B, Q, R)

def predict_statedot_given_observation_and_action(x, xdot, theta, thetadot, u):
    return np.dot(A, [x, xdot, theta, thetadot]) + np.dot(B, u)

done = False
x, xdot, theta, thetadot = env.reset()
env.state = np.array([0, 0, 1 * np.pi / 180, 0])
x, xdot, theta, thetadot = env.state
while True:
    print('state: ', [x, xdot, theta, thetadot])

    u = -np.dot(K, [x, xdot, theta, thetadot])[0]
    print('u: ', u)
    print('xpred: ', predict_statedot_given_observation_and_action(x, xdot, theta, thetadot, u))

    if u > 0.0:
      action = 1
    else:
      action = 0

    state, reward, done, info = env.step(action)
    x, xdot, theta, thetadot = state

    env.render()

    if done:
        import ipdb; ipdb.set_trace()
        x, xdot, theta, thetadot = env.reset()