import sys
import numpy as np
import matplotlib.pyplot as plt


class Window:
    """Windows to draw a girdworld instance using Matplotlib"""

    def __init__(self, title):
        self.fig = None
        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img):
        """Show an image or update the image being shown"""

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_caption(self, text):
        """Set/update the caption text below the image"""
        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """Register a keyboard event handler"""
        self.fig.canvas.mol_connect('key_press_event', key_handler)

    def show(self, block=True):
        """Show the window, and sstart an event loop"""

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interactive mode, this enters the matplotlib event loop
        plt.show()

    def close(self):
        """Close the window"""
        plt.close()