
import numpy as np
from numpy.typing import NDArray

from robot.settings import ActionOption
from robot.move_calculator import circle_data
from robot.move_calculator import rectangular_data
from robot.move_calculator import position_from_pose
from robot.move_calculator import rotate_pose


class Agent:
    def __init__(self):
        self.set_action(ActionOption.circle)
        self.set_search_area(0.5, np.pi / 3.)
        self.set_search_noise(sigma=0.1)

    def set_search_area(self, length: float, angle: float) -> None:
        self._search_length = length
        self._search_angle = angle

    def set_search_noise(self, sigma: float) -> None:
        self._search_noise_sigma = sigma

    def set_action(self, action_option: ActionOption) -> None:
        self._action_option = action_option
        self._calculate_poses()

    def _calculate_poses(self) -> None:
        match self._action_option:
            case ActionOption.circle:
                self._poses = circle_data(radius=1.5, number=100, start=0., end=2*np.pi)
            case ActionOption.rectangle:
                self._poses = rectangular_data(good_data=True)
            case _:
                self._poses = circle_data(radius=1.5, number=100, start=0., end=2*np.pi)
        # reset index and define data length
        self._index = 0
        self._data_length = len(self._poses)

    def next_pose(self) -> NDArray:
        if self._index < self._data_length - 1:
            self._index += 1
        else:
            self._index = 0
        return self.get_pose()

    def get_pose(self) -> NDArray:
        return self._poses[self._index]

    def observe(self, landmarks: NDArray) -> NDArray:
        search_points = self._calculate_search_area()
        for landmark in landmarks:
            is_in_search_area = self._point_in_the_area(landmark, search_points)
        return np.array([])

    def _calculate_search_area(self) -> NDArray:
        pose = self.get_pose()
        pt1 = position_from_pose(pose)
        pt2 = position_from_pose(rotate_pose(pose, self._search_angle / 2.))
        pt3 = position_from_pose(rotate_pose(pose, -self._search_angle / 2.))
        return np.c_[pt1, pt2, pt3]

    def _point_in_the_area(self, point: NDArray, areas: NDArray) -> bool:
        return False
