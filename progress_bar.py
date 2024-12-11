import time
from logger import mylog

logger = mylog.get_logger()

def human_readable_time(seconds):
    # Simple placeholder; adjust formatting as desired.
    # Example: convert seconds to H:MM:SS
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    else:
        return f"{secs}s"


def print_progress_bar(name: str, processed_count: int, total_units: int, elapsed_secs: float):
    units_left = total_units - processed_count
    progress_fraction = processed_count / total_units
    remaining_time = (elapsed_secs / processed_count) * units_left if processed_count > 0 else 0
    batch_duration = elapsed_secs

    # Build the progress bar
    bar_length = 50
    filled_length = int(bar_length * progress_fraction)

    # Choose bar color: blue in-progress, green when complete
    if processed_count < total_units:
        bar_color = "\033[94m"  # Blue
    else:
        bar_color = "\033[92m"  # Green

    bar = bar_color + '#' * filled_length + '-' * (bar_length - filled_length) + "\033[0m"

    # Highlight the remaining time in yellow
    remaining_time_str = f"{human_readable_time(remaining_time)}"
    colored_remaining_time = f"\033[93m{remaining_time_str}\033[0m"

    # Print the progress line, overwriting the previous line
    print(
        f"\r{name}: |{bar}| {processed_count}/{total_units} "
        f"Elapsed: {human_readable_time(batch_duration)} "
        f"Predictions left: {units_left} "
        f"Est. remaining: {colored_remaining_time}",
        end='',
        flush=True
    )

def progr(name: str, total_units: int):
    start_time = time.time()
    for processed_count in range(1, total_units + 1):
        # Simulate work
        time.sleep(0.05)
        print_progress_bar(name, processed_count, total_units, time.time() - start_time)



    # Move to a new line after completion
    print("\nDone!")
