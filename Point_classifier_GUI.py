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

def on_get_points(red_points, blue_points, plot, fig):
    r_ar = np.array(red_points)
    b_ar = np.array(blue_points)
    X = np.r_[r_ar, b_ar]



    y_r = np.zeros(np.shape(r_ar)[0])
    y_b = np.ones(np.shape(b_ar)[0])
    y = np.r_[y_r, y_b]



    print(f'x: {X}')
    print(f'X[:,0]: {X[:,0]}')
    print(f'y: {y}')
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
    print(f'X_pts = {X_pts}')
    # Generate random z values for demonstration purposes
    z_pts = np.random.random(len(X_pts))

    y_inv = [0]*len(y)
    for i in range(len(y)):
        if y[i] == 0:
            y_inv[i] = 1
    y_train = np.c_[y,y_inv]
    # treinar a rede aqui
    clf = pcl.create_new_classifier()
    pcl.train_neural_network(clf, X, y_train)

    for i in range(0,len(X_pts)):

        z_pts[i] = clf.predict(X_pts[i])[0]
    # Plot color map
    # plt.tripcolor(X_pts[:, 0], X_pts[:, 1], z_pts, cmap='magma')

    # Plot red and blue points
    red_points = np.array(red_points)
    blue_points = np.array(blue_points)
    # Clear the existing plot
    plot.clear()

     # Plot red and blue points
    plot.plot(red_points[:, 0], red_points[:, 1], 'ro')
    plot.plot(blue_points[:, 0], blue_points[:, 1], 'bo')

    # Create a Rectangle patch
    # Example usage
    value = 0.5
    min_value = 0.0
    max_value = 1.0
    min_color = (240, 0, 0)  # Red
    max_color = (0, 0, 240)  # Blue


    # print(color_code)  # Output: #7f007f
    for i in range(0,np.shape(X_pts)[0]):
        #usar o predict jogando os valores de X_pts
        color_code = get_color_code(z_pts[i], min_value, max_value, min_color, max_color)
        rect = patches.Rectangle(xy=(X_pts[i][0], X_pts[i][1]), width=square_side_x, height=square_side_y, linewidth=0, edgecolor=color_code, facecolor=color_code)
        # Add the patch to the Axes
        plot.add_patch(rect)



    # Redraw the canvas
    fig.canvas.draw_idle()

    # print(f'x_test = {X_pts}')
    # plt.imshow(X_pts, aspect='auto', cmap=plt.get_cmap('magma'), origin='lower')
def draw_result(X_pts, y_pts):
    pass
def main():
    red_points = []
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

    root = tk.Tk()
    root.title("Point Drawer")

    # Create a frame for the buttons
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # Create buttons in the button frame
    button1 = tk.Button(button_frame, text="Red", command=partial(on_bt_red, current_color))
    button1.pack(side=tk.BOTTOM, pady=10)

    button2 = tk.Button(button_frame, text="Button 2", command=partial(on_bt_blue, current_color))
    button2.pack(side=tk.BOTTOM, pady=10)



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

    button3 = tk.Button(button_frame, text="Button 3", command=partial(on_get_points, red_points, blue_points, plot, fig))
    button3.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()

def get_color_code(value, min_value, max_value, min_color, max_color):
    # Clamp the value within the specified range
    value = max(min(value, max_value), min_value)

    # Calculate the normalized position of the value within the range
    normalized_value = (value - min_value) / (max_value - min_value)

    # Interpolate the RGB values between the minimum and maximum colors
    r = int((max_color[0] - min_color[0]) * normalized_value + min_color[0])
    g = int((max_color[1] - min_color[1]) * normalized_value + min_color[1])
    b = int((max_color[2] - min_color[2]) * normalized_value + min_color[2])

    # Return the color code as a string
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == '__main__':
    main()
