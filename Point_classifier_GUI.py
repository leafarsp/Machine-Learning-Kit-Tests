import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_interactions import ioff, panhandler, zoom_factory
from functools import partial
import numpy as np



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

def on_get_points(red_points, blue_points):
    r_ar = np.array(red_points)
    b_ar = np.array(blue_points)
    X = np.r_[r_ar, b_ar]
    y_r = np.zeros(np.shape(r_ar)[0])
    y_b = np.ones(np.shape(b_ar)[0])
    y = np.r_[y_r, y_b]
    print(f'x: {X}')
    print(f'X[:,0]: {X[:,0]}')
    print(f'y: {y}')
    lengh_Xx = np.max(X[:,0])-np.min(X[:,0])
    lengh_Xy = np.max(X[:, 1]) - np.min(X[:, 1])

    center_Xx = np.min(X[:,0]) + lengh_Xx/2
    center_Xy = np.min(X[:, 1]) + lengh_Xy / 2

    print(f'lengh_Xx = {lengh_Xx}, lengh_Xy = {lengh_Xy}')
    test_points_Xx = 10
    test_points_Xy = 10
    x_test_min = center_Xx - lengh_Xx * 1,2
    x_test_max = center_Xx + lengh_Xx * 1, 2
    y_test_min = center_Xy - lengh_Xy * 1, 2
    y_test_max = center_Xy + lengh_Xy * 1, 2
    x_test = list(np.linspace(x_test_min, x_test_max, test_points_Xx-1))
    y_test = list(np.linspace(y_test_min, y_test_max, test_points_Xy - 1))
    z_test = list(np.linspace(0,1,2))
    xv, yv = np.meshgrid(x_test,y_test)
    X_pts = np.stack((np.ravel(xv), np.ravel(yv)),axis=-1)

    # Plot color map
    plt.imshow(X_pts[..., 2].reshape(xv.shape), aspect='auto', cmap='magma',
               origin='lower', extent=[x_test_min, x_test_max, y_test_min, y_test_max], alpha=0.5)

    # Plot red and blue points
    red_points = np.array(red_points)
    blue_points = np.array(blue_points)
    plt.plot(red_points[:, 0], red_points[:, 1], 'ro')
    plt.plot(blue_points[:, 0], blue_points[:, 1], 'bo')

    plt.show()

    print(f'x_test = {X_pts}')
    # plt.imshow(X_pts, aspect='auto', cmap=plt.get_cmap('magma'), origin='lower')

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
    button1 = tk.Button(button_frame, text="Button 1", command=partial(on_bt_red, current_color))
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

    button3 = tk.Button(button_frame, text="Button 3", command=partial(on_get_points, red_points, blue_points))
    button3.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()



if __name__ == '__main__':
    main()
