import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_interactions import ioff, panhandler, zoom_factory
from functools import partial
import numpy as np
import Point_classifier_MLK as pcl

class DefineColor:
    def __init__(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

def on_bt_red(current_color: DefineColor):
    current_color.color = 0

def on_bt_blue(current_color: DefineColor):
    current_color.color = 1

def on_bt_green(current_color: DefineColor):
    current_color.color = 2

def on_get_points(red_points, green_points, blue_points, plot, fig):
    r_ar = np.array(red_points)
    g_ar = np.array(green_points)
    b_ar = np.array(blue_points)
    X = np.r_[r_ar, g_ar, b_ar]

    y_r = [[1., 0., 0.]] * len(r_ar)
    y_b = [[0., 1., 0.]] * len(g_ar)
    y_g = [[0., 0., 1.]] * len(b_ar)
    y = np.r_[y_r, y_g, y_b]




    lengh_Xx = np.abs(np.max(X[:, 0]) - np.min(X[:, 0]))
    lengh_Xy = np.abs(np.max(X[:, 1]) - np.min(X[:, 1]))

    center_Xx = np.min(X[:, 0]) + lengh_Xx / 2
    center_Xy = np.min(X[:, 1]) + lengh_Xy / 2

    print(f'lengh_Xx = {lengh_Xx}, lengh_Xy = {lengh_Xy}')
    print(f'center_Xx = {center_Xx}, center_Xy = {center_Xy}')
    test_points_Xx = 50
    test_points_Xy = 50

    x_test_min = center_Xx - lengh_Xx * 1.2 / 2.
    x_test_max = center_Xx + lengh_Xx * 1.2 / 2.

    y_test_min = center_Xy - lengh_Xy * 1.2 / 2.
    y_test_max = center_Xy + lengh_Xy * 1.2 / 2.

    x_test = list(np.linspace(x_test_min, x_test_max, test_points_Xx+1))
    y_test = list(np.linspace(y_test_min, y_test_max, test_points_Xy+1))
    z_test = list(np.linspace(0,1,2))
    square_side_x = (lengh_Xx * 1.2) / (test_points_Xx)
    square_side_y = (lengh_Xx * 1.2) / (test_points_Xy)

    xv, yv = np.meshgrid(x_test,y_test)
    X_pts = np.stack((np.ravel(xv), np.ravel(yv)),axis=-1)
    # z_pts = [[0, 0, 0]] * test_points_Xx * test_points_Xy
    # print(f'X_pts = {X_pts}')
    # Generate random z values for demonstration purposes
    color_pts = np.random.random((len(X_pts),3))
    #
    # y_inv = [0]*len(y)
    # for i in range(len(y)):
    #     if y[i] == 0:
    #         y_inv[i] = 1
    # y_train = np.c_[y,y_inv]
    # treinar a rede aqui
    clf = pcl.create_new_classifier()
    clf.max_epoch_sprint = 100
    for _ in range(0,10000):
        if clf.state == 'training_finished':
            break
        pcl.train_neural_network(clf, X, y)
        clf.max_epoch_sprint = clf.t + 100

        for i in range(0,len(X_pts)):
            color_pts[i] = clf.predict(X_pts[i])

        draw_result(fig, plot, X_pts, color_pts, red_points, green_points, blue_points, square_side_x, square_side_y)


def draw_result(fig, plot, X_pts, color_pts, red_points, green_points, blue_points, square_side_x, square_side_y):
    red_points = np.array(red_points)
    blue_points = np.array(blue_points)
    green_points = np.array(green_points)
    # Clear the existing plot
    plot.clear()

    # Plot red and blue points
    plot.plot(red_points[:, 0], red_points[:, 1], 'ro')
    plot.plot(blue_points[:, 0], blue_points[:, 1], 'bo')
    plot.plot(green_points[:, 0], green_points[:, 1], 'go')
    # Create a Rectangle patch
    # Example usage
    value = 0.5
    min_value = -1.0
    max_value = 1.0
    min_color = 0  # Red
    max_color = 240  # Blue

    # print(color_code)  # Output: #7f007f
    for i in range(0, np.shape(X_pts)[0]):
        # usar o predict jogando os valores de X_pts
        color_code = get_color_code(color_pts[i], min_value, max_value, min_color, max_color)

        rect = patches.Rectangle(xy=(X_pts[i][0], X_pts[i][1]), width=square_side_x, height=square_side_y, linewidth=0,
                                 edgecolor=color_code, facecolor=color_code)
        # Add the patch to the Axes
        plot.add_patch(rect)

    # Redraw the canvas
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def get_color_code(color_pts:list, min_value, max_value, min_color, max_color):
    # Clamp the value within the specified range
    r_value = max(min(color_pts[0], max_value), min_value)
    g_value = max(min(color_pts[1], max_value), min_value)
    b_value = max(min(color_pts[2], max_value), min_value)

    # Calculate the normalized position of the value within the range
    r_normalized_value = (r_value - min_value) / (max_value - min_value)
    g_normalized_value = (g_value - min_value) / (max_value - min_value)
    b_normalized_value = (b_value - min_value) / (max_value - min_value)

    # Interpolate the RGB values between the minimum and maximum colors
    r = int((max_color - min_color) * r_normalized_value + min_color)
    g = int((max_color - min_color) * g_normalized_value + min_color)
    b = int((max_color - min_color) * b_normalized_value + min_color)

    # Return the color code as a string
    return f"#{r:02x}{g:02x}{b:02x}"
def main():
    red_points = []
    green_points = []
    blue_points = []
    red_color_activated = []
    blue_color_activated = []
    current_color = DefineColor(0)

    def on_click(event):
        if event.button == 1:
            if current_color.color == 0:
                red_points.append((event.xdata, event.ydata))
                plot.plot(event.xdata, event.ydata, 'ro')
                fig.canvas.draw_idle()
            elif current_color.color == 1:
                blue_points.append((event.xdata, event.ydata))
                plot.plot(event.xdata, event.ydata, 'bo')
                fig.canvas.draw_idle()
            elif current_color.color == 2:
                green_points.append((event.xdata, event.ydata))
                plot.plot(event.xdata, event.ydata, 'go')
                fig.canvas.draw_idle()

    root = tk.Tk()
    root.title("Point Classifier")

    # Create a frame for the buttons
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # Create buttons in the button frame
    button1 = tk.Button(button_frame, text="Red", command=partial(on_bt_red, current_color))
    button1.pack(side=tk.BOTTOM, pady=10)

    button2 = tk.Button(button_frame, text="Blue", command=partial(on_bt_blue, current_color))
    button2.pack(side=tk.BOTTOM, pady=10)

    button4 = tk.Button(button_frame, text="Green", command=partial(on_bt_green, current_color))
    button4.pack(side=tk.BOTTOM, pady=10)


    # Create a frame for the plot window
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    # Create a figure and canvas for the plot window
    fig = Figure(figsize=(5, 5), dpi=100)
    plot = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Enable zooming with the scroll wheel
    zoom_factory(plot, base_scale=1.1)

    # Bind the click event to the plot canvas
    canvas.mpl_connect('button_press_event', on_click)

    button3 = tk.Button(button_frame, text="Train network", command=partial(on_get_points, red_points, green_points,
                                                                            blue_points, plot, fig))
    button3.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()




if __name__ == '__main__':
    main()
