from route_optimizer_functions import default_problem_data, run_optimizer


def main():
    data = default_problem_data()

    # Optional customisation examples:
    # data["n_slots"] = 3
    # data["season"] = "winter"
    # data["user_gear"] = data["gear_level"]["Trail"]

    run_optimizer(data=data, reps=2, maxiter=300)


if __name__ == "__main__":
    main()
