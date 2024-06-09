
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time
from numpy.typing import NDArray

from robot.agent import Agent
from robot.settings import ActionOption
from robot.util import calculate_transformation_between_points


# global plots
fig, axes = plt.subplots(2, 1, figsize=(8, 16))
axes[0].set_title('World View')
axes[1].set_title('Robot View')


# set x,y lims
def set_xy_limits() -> None:
    global axes
    global AREA_X
    global AREA_Y
    axes[0].set_xlim(AREA_X)
    axes[0].set_ylim(AREA_Y)
    axes[1].set_xlim([1. * x for x in AREA_X])
    axes[1].set_ylim([1. * y for y in AREA_Y])


# generate landmarks
def generate_landmarks(number: int, x_range: list[float], y_range: list[float]) -> NDArray:
    xs = np.random.random(number) * (x_range[1] - x_range[0]) + x_range[0]
    ys = np.random.random(number) * (y_range[1] - y_range[0]) + y_range[0]
    return np.c_[xs, ys]


# add pose vector
def draw_pose(plots: list, pose: NDArray, length: float) -> None:
    xy = pose[:-1, -1]
    vec1 = pose[:-1, 0]
    vec2 = pose[:-1, 1]
    x_line, y_line = plots[0], plots[1]
    x_line.set_xdata([xy[0], xy[0] + length * vec1[0]])
    x_line.set_ydata([xy[1], xy[1] + length * vec1[1]])
    y_line.set_xdata([xy[0], xy[0] + length * vec2[0]])
    y_line.set_ydata([xy[1], xy[1] + length * vec2[1]])


# draw trajectory
def draw_trajectory(plot, xys: list[list[float]]) -> None:
    plot.set_xdata(xys[0])
    plot.set_ydata(xys[1])


# global matplotplot
AREA_X = [-1.5, 1.5]
AREA_Y = [-1.5, 1.5]
POSE_LENGTH = 0.1
TRAJECTORY_SIZE = 30

# world view
map_scatter = axes[0].scatter([], [], c='black', s=20)
gt_x_line, = axes[0].plot([], [], c='gray')
gt_y_line, = axes[0].plot([], [], c='pink')
predicted_x_line, = axes[0].plot([], [], c='black')
predicted_y_line, = axes[0].plot([], [], c='red')
trajectory_gt, = axes[0].plot([], [], c='C0')
trajectory_predicted, = axes[0].plot([], [], c='C1')

# robot view
robot_x_line, = axes[1].plot([0, 0], [0, POSE_LENGTH], c='black')
robot_y_line, = axes[1].plot([0, -POSE_LENGTH], [0, 0], c='red')
observed_scatter = axes[1].scatter([], [], c='black', s=20)

# initialize xy axes
set_xy_limits()


class StreamlitViewer:
    def __init__(self):
        self._title = 'Robot Simple Viewer'
        self._sidebar_title = 'Setting'

    def display_title(self):
        st.title(self._title)

    def display_sidebar(self):
        st.sidebar.title(self._sidebar_title)
        self._movement_option = st.sidebar.selectbox('Select Movement', ('Circle', 'Rectangle'))
        self._landmark_number = st.sidebar.number_input(
            '# Landmarks', min_value=3, max_value=30, value=20, step=1)

    def display_main_content(self):
        st.write('Welcome to the Robot Localization Viewer!')

        global fig
        global map_scatter

        self.plot_area = st.empty()
        self.st_plot = self.plot_area.pyplot(fig)

        # get settings
        self._map_data = generate_landmarks(self._landmark_number, AREA_X, AREA_Y)
        action_option = ActionOption.circle if self._movement_option == 'Circle' \
            else ActionOption.rectangle

        # generate robot
        self.agent = Agent()
        self.agent.set_map(self._map_data)
        self.agent.set_action(action_option)
        self.agent.set_observing_noise(sigma=0.1)

        self._map_color = np.arange(len(self._map_data))

        # map scatter
        map_scatter.set_offsets(self._map_data)
        map_scatter.set_array(self._map_color)

        # trajectories
        self._traj_gt = [[], []]
        self._traj_predicted = [[], []]

        while True:
            self.animate()
            time.sleep(0.01)

    def animate(self) -> None:
        global fig
        global observed_scatter
        global gt_x_line, gt_y_line
        global trajectory_gt, trajectory_predicted

        pose = self.agent.next_pose()
        observed = self.agent.observe()
        solved_pose = calculate_transformation_between_points(
            self.agent.get_map()[:, :-1], observed)

        # trajectories
        if len(self._traj_gt[0]) >= TRAJECTORY_SIZE:
            self._traj_gt[0].pop(0)
            self._traj_gt[1].pop(0)
        if len(self._traj_predicted[0]) >= TRAJECTORY_SIZE:
            self._traj_predicted[0].pop(0)
            self._traj_predicted[1].pop(0)
        self._traj_gt[0].append(pose[0, -1])
        self._traj_gt[1].append(pose[1, -1])
        self._traj_predicted[0].append(solved_pose[0, -1])
        self._traj_predicted[1].append(solved_pose[1, -1])

        # world view
        draw_pose([gt_x_line, gt_y_line], pose, POSE_LENGTH)
        draw_pose([predicted_x_line, predicted_y_line], solved_pose, POSE_LENGTH)
        draw_trajectory(trajectory_gt, self._traj_gt)
        draw_trajectory(trajectory_predicted, self._traj_predicted)

        # robot view
        observed_x = -observed[:, 1]
        observed_y = observed[:, 0]
        observed_xy = np.c_[observed_x, observed_y]
        observed_scatter.set_offsets(observed_xy)
        observed_scatter.set_array(self._map_color)

        set_xy_limits()
        self.st_plot.pyplot(fig)

    def run(self):
        self.display_title()
        self.display_sidebar()
        self.display_main_content()


if __name__ == '__main__':
    viewer = StreamlitViewer()
    viewer.run()
