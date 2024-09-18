import numpy as np
import matplotlib.pyplot as plt

# Define the function
def wave_pulse(x):
    return np.sin(20 * x) * np.exp(-x**2)

def main():
    # Generate x values
    x = np.linspace(-3, 3, 1000)

    # Compute the corresponding y values
    y = wave_pulse(x)

    # Create the plot
    plt.plot(x, y, color='green')  # Set the plot color to green

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of f(x) = sin(20x) * e^(-x^2)')

    plt.xlim(3, -3)
    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()