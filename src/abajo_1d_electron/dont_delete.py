# plots_data = [
#         {
#             "input_func": lambda x: 2 * x / R_max,
#             "title": r"$\frac{2x}{R_{max}}$"
#         },
#         {
#             "input_func": lambda x: -2 * x / R_max,
#             "title": r"$-\frac{2x}{R_{max}}$"
#         },
#         {
#             "input_func": lambda x: 2 * x / R_max * np.sin(4 * np.pi * x / R_max),
#             "title": r"$\frac{2x}{R_{max}} \cdot \sin\left(\frac{20\pi x}{R_{max}}\right)$"
#         },
#         {
#             "input_func": lambda x: 2 * (x + move_by) / R_max * np.sin(4 * np.pi * (x + move_by) / R_max) + 0.2,
#             "title": "asymmetric sin"
#         }
#     ]