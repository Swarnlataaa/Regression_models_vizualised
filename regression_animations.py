from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class RegressionScene(Scene):
    def construct(self):
        # Create data points
        np.random.seed(0)
        X = np.linspace(-5, 5, 10).reshape(-1, 1)
        y = 2 * X + np.random.normal(0, 1, (10, 1))

        # Create axes
        axes = Axes(
            x_range=(-6, 6, 1),
            y_range=(-10, 10, 1),
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "color": WHITE},
            x_axis_config={"numbers_with_elongated_ticks": range(-5, 6)},
            y_axis_config={"numbers_with_elongated_ticks": range(-9, 10, 2)},
        )

        # Create animation objects
        text_objects = [
            Text("Linear Regression"),
            Text("Decision Tree Regression"),
            Text("Random Forest Regression"),
            Text("K-Nearest Neighbors Regression"),
            Text("Support Vector Regression"),
        ]
        visual_objects = [
            self.create_regression_visual(X, y, axes, LinearRegression()),
            self.create_regression_visual(X, y, axes, DecisionTreeRegressor()),
            self.create_regression_visual(X, y, axes, RandomForestRegressor()),
            self.create_regression_visual(X, y, axes, KNeighborsRegressor()),
            self.create_regression_visual(X, y, axes, SVR()),
        ]

        # Position the text and visual objects
        for text_obj, visual_obj in zip(text_objects, visual_objects):
            text_obj.to_edge(UP)
            visual_obj.next_to(text_obj, DOWN, buff=0.5)
            self.play(Write(text_obj))
            self.play(Create(visual_obj), run_time=2)
            self.wait(2)
            self.play(FadeOut(text_obj), FadeOut(visual_obj), run_time=1)
            self.wait(0.5)

        # Remove axes and show "Thank you"
        self.play(FadeOut(axes))
        thank_you_text = Text("Thank you!", font_size=48).set_color(YELLOW)
        self.play(Write(thank_you_text))
        self.wait(2)

        # Add footer with copyright notice
        footer_text = Text("Copyright © C Swarnlata", font_size=24).set_color(GRAY)
        footer_text.to_corner(DL, buff=0.5)
        self.play(Write(footer_text))
        self.wait(2)

    def create_regression_visual(self, X, y, axes, regression_model):
        regression_model.fit(X, y)
        y_pred = regression_model.predict(X)

        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(np.minimum(y, y_pred)), np.max(np.maximum(y, y_pred))

        points = self.create_points(X, y_pred, axes)
        lines = self.create_lines(points)
        points.move_to(axes.c2p(0, 0))
        lines.move_to(axes.c2p(0, 0))
        axes.x_range = (x_min, x_max)
        axes.y_range = (y_min, y_max)
        return VGroup(axes, points, lines)

    def create_points(self, X, y, axes):
        points = VGroup()
        for x_val, y_val in zip(X.flatten(), y.flatten()):
            point = Dot(axes.c2p(x_val, y_val))
            points.add(point)
        return points

    def create_lines(self, points):
        lines = VGroup()
        for start, end in zip(points[:-1], points[1:]):
            line = Line(start.get_center(), end.get_center(), stroke_width=2)
            lines.add(line)
        return lines


if __name__ == "__main__":
    regression_scene = RegressionScene()
    regression_scene.render()
