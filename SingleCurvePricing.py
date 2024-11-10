import numpy as np
import matplotlib.pyplot as plt


def interpolate_zero_rate(t, maturities, swap_rates):
    """
    Linearly interpolation of log discount factors from market swap rates.

        t: The time (maturity) for which to interpolate the discount factor.
        maturities: Maturities of the swaps.
        swap_rates: Market swap rates corresponding to the maturities.
    """
    # Interpolating zero rates for time t
    df = np.array(np.exp(-swap_rates * maturities))
    log_df = np.log(df)
    interp_log_df = np.interp(t, maturities, log_df)
    interp_df = np.exp(interp_log_df)
    return interp_df


def analytic_delta(notional, start, maturities, freq, swap_rate):
    """
    Calculation of DV01 of a swap with respect to the fixed (swap) rate
    """
    # Create a time grid of cashflows
    time_grid = [
        np.arange(start + freq, maturities[i] + freq, freq)
        for i in range(len(maturities))
    ]
    tau = freq
    delta = []

    for time in time_grid:
        d = 0  # Reset d for each swap
        for t in time:
            d += tau * interpolate_zero_rate(t, maturities, swap_rate)
        delta.append(d)

    delta = np.array(delta) * notional
    return delta


def single_curve(notional, start, end, freq, swap_rate, swap_type, discount_rate):
    """
    Calculate the present value of a swap using the discount curve.

        notional: Notional amount of the swap.
        start: Start time of the swap.
        end: End time of the swap.
        maturity: Maturity of the swap.
        freq: Frequency of payments (e.g., 0.5 for semi-annual).
        swap_rate: Swap rate.
        swap_type: +1 for receiving fixed, -1 for paying fixed.
        discount_curve: Function that returns discount factors given time.
    """
    fixed_leg_pv = 0.0
    float_leg_pv = 0.0

    # Create time grid for payments
    time_grid = np.arange(start + freq, end + freq, freq)
    tau = freq

    discount_start = 1
    discount_end = discount_rate(end)

    # Calculate PV of the fixed leg
    for time in time_grid:
        fixed_leg_pv += swap_rate * tau * discount_rate(time)

    # Calculate PV of the floating leg
    float_leg_pv = discount_start - discount_end

    # Net PV based on swap type
    net_pv = (fixed_leg_pv - float_leg_pv) * swap_type

    return net_pv * notional


def multivariate_newton_raphson(
    zero_rates, maturities, instruments, max_iterations=100, tolerance=1e-16
):
    """
    Apply Multivariate Newton-Raphson method to calibrate zero rates to match instrument prices.

        zero_rates: Initial guess for zero rates.
        maturities: Maturities of the instruments.
        instruments: List of swap pricing functions.
        max_iterations: Maximum number of iterations to run.
        tolerance: Convergence threshold for the error.
    """
    error_norm = 10e10  # Large initial error
    iteration = 0

    while error_norm > tolerance and iteration < max_iterations:
        iteration += 1
        values = evaluate_instruments(zero_rates, maturities, instruments)
        jacobian_matrix = compute_jacobian(zero_rates, maturities, instruments)
        jacobian_inv = np.linalg.inv(jacobian_matrix)
        error_norm = -np.dot(jacobian_inv, values)
        zero_rates = zero_rates + error_norm
        error_norm = np.linalg.norm(error_norm)
        print(f"Iteration {iteration}: Error Norm = {error_norm:.5e}")

    return zero_rates


def compute_jacobian(zero_rates, maturities, instruments, epsilon=1e-5):
    """
    Compute the Jacobian matrix for the system of equations.

        zero_rates: Initial guess.
        maturities: Maturities of the instruments.
        instruments: List of swap pricing functions.
        epsilon: Small perturbation for numerical differentiation.
    """
    num_instruments = len(maturities)
    jacobian_matrix = np.zeros([num_instruments, num_instruments])

    base_values = evaluate_instruments(zero_rates, maturities, instruments)
    perturbed_rates = np.copy(zero_rates)

    for i in range(len(zero_rates)):
        perturbed_rates[i] = zero_rates[i] + epsilon
        perturbed_values = evaluate_instruments(
            perturbed_rates, maturities, instruments
        )
        perturbed_rates[i] = zero_rates[i]

        # Compute the partial derivative (finite difference)
        jacobian_matrix[:, i] = (perturbed_values - base_values) / epsilon

    return jacobian_matrix


def evaluate_instruments(swap_rates, maturities, instruments):
    """
    Evaluate the prices of instruments using the current discount curve.

        swap_rates: Market rates.
        maturities: Maturities of the instruments.
        instruments: List of swap pricing functions.
    """
    values = np.zeros([len(maturities)])
    discount_rate = lambda t: interpolate_zero_rate(t, maturities, swap_rates)
    for i in range(len(maturities)):
        values[i] = instruments[i](discount_rate)
    return values


def main():
    maturities = np.array([1, 2, 3, 5, 7, 10, 15, 30])

    swap_rates = np.array(
        [0.04333, 0.04032, 0.03914, 0.03796, 0.03761, 0.03794, 0.03895, 0.03919]
    )

    # Initial guess for zero rates
    initial_rates = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])

    # Define swap instruments
    swap1 = lambda discount_factor: single_curve(
        1, 0, maturities[0], 0.25, swap_rates[0], -1, discount_factor
    )
    swap2 = lambda discount_factor: single_curve(
        1, 0, maturities[1], 0.25, swap_rates[1], -1, discount_factor
    )
    swap3 = lambda discount_factor: single_curve(
        1, 0, maturities[2], 0.25, swap_rates[2], -1, discount_factor
    )
    swap4 = lambda discount_factor: single_curve(
        1, 0, maturities[3], 0.25, swap_rates[3], -1, discount_factor
    )
    swap5 = lambda discount_factor: single_curve(
        1, 0, maturities[4], 0.25, swap_rates[4], -1, discount_factor
    )
    swap6 = lambda discount_factor: single_curve(
        1, 0, maturities[5], 0.25, swap_rates[5], -1, discount_factor
    )
    swap7 = lambda discount_factor: single_curve(
        1, 0, maturities[6], 0.25, swap_rates[6], -1, discount_factor
    )
    swap8 = lambda discount_factor: single_curve(
        1, 0, maturities[7], 0.25, swap_rates[7], -1, discount_factor
    )
    instruments = [swap1, swap2, swap3, swap4, swap5, swap6, swap7, swap8]

    swaps_initial = evaluate_instruments(initial_rates, maturities, instruments)

    print(f"Initial swap values = {swaps_initial}")

    # Run Newton-Raphson method to calibrate zero rates
    optimized_rates = multivariate_newton_raphson(
        initial_rates, maturities, instruments
    )
    print(f"Optimised Zero Rates: {optimized_rates}")

    swaps_final = evaluate_instruments(optimized_rates, maturities, instruments)

    print(f"Optimised swap values = {swaps_final}")

    market_deltas = analytic_delta(1, 0, maturities, 0.25, swap_rates)
    optimised_deltas = analytic_delta(1, 0, maturities, 0.25, optimized_rates)

    print(f"Market deltas = {market_deltas}")
    print(f"Optimised deltas = {optimised_deltas}")
    print(f"Differences in deltas = {optimised_deltas - market_deltas}")

    # Plot the initial, market rates and optimized curves
    plt.plot(
        maturities,
        initial_rates,
        color="red",
        marker="o",
        linestyle="",
        label="Initial guess",
    )
    plt.plot(
        maturities,
        swap_rates,
        color="black",
        marker="x",
        linestyle="solid",
        label="Market Zero Rates",
    )
    plt.plot(
        maturities,
        optimized_rates,
        color="green",
        marker="^",
        linestyle="solid",
        label="Optimized Zero Rates",
    )
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero Rates")
    plt.grid()
    plt.legend()
    plt.show()


main()
