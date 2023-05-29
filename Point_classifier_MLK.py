import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial
import numpy as np




class DefineColor:
    def __init__(self,color):
        self.color = color

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color




def on_bt_red(current_color: DefineColor):
    current_color.color = 0


def on_bt_blue(current_color: DefineColor):
    current_color.color = 1

def zoom(event):
    ax = plt.gca()
    if event.button == 'up':
        ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
        ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
    elif event.button == 'down':
        ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
        ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
    fig.canvas.draw()



def on_get_points(red_points,blue_points):
    # print("Red points:", red_points)
    # print("Blue points:", blue_points)
    r_ar = np.array(red_points)
    b_ar = np.array(blue_points)
    x = np.r_[r_ar, b_ar]
    y_r = np.zeros(np.shape(r_ar)[0])
    y_b = np.ones(np.shape(b_ar)[0])
    y = np.r_[y_r, y_b]
    print(f'x: {x}')
    print(f'y: {y}')





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
                plt.plot(event.xdata, event.ydata, 'ro')
                fig.canvas.draw_idle()
            elif current_color.color == 1:
                blue_points.append((event.xdata, event.ydata))
                plt.plot(event.xdata, event.ydata, 'bo')
                fig.canvas.draw_idle()

    def zoom_in():
        # ax = plt.gca()
        pass
        # ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
        # ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
        # fig.canvas.draw()

    def zoom_out():
        pass
        # ax = plt.gca()
        #
        # ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
        # ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        # fig.canvas.draw()

    root = tk.Tk()
    root.title("Point Drawer")

    # Create a frame for the buttons
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # Create buttons in the button frame
    button1 = tk.Button(button_frame, text="Button 1", command=partial(on_bt_red,current_color))
    button1.pack(side=tk.BOTTOM, pady=10)

    button2 = tk.Button(button_frame, text="Button 2", command=partial(on_bt_blue,current_color))
    button2.pack(side=tk.BOTTOM, pady=10)

    button3 = tk.Button(button_frame, text="Button 3", command=partial(on_get_points,red_points,blue_points))
    button3.pack(side=tk.BOTTOM, pady=10)

    button_zoom_in = tk.Button(button_frame, text="Zoom In", command=zoom_in)
    button_zoom_in.pack(side=tk.BOTTOM, pady=10)

    button_zoom_out = tk.Button(button_frame, text="Zoom Out", command=zoom_out)
    button_zoom_out.pack(side=tk.BOTTOM, pady=10)

    # Create a frame for the plot window
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    # Create a figure and canvas for the plot window
    fig = Figure(figsize=(5, 5), dpi=100)
    plt = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Bind the click event to the plot canvas
    canvas.mpl_connect('button_press_event', on_click)

    root.mainloop()



if __name__ == '__main__':
    main()
