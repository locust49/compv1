import argparse
import re


# def get_polynom_regex() -> list[str]:
# 	"""
# 	:return: list of regexes for polynom
# 	"""
# 	return [r"(-?\d+\.\d+|-?\d+)?x\^(-?\d+\.\d+|-?(\d+))",
# 									r"(-?\d+\.\d+|-?\d+)x",
# 									r"(-?\d+\.\d+|-?\d+)"]


def get_polynom_regex() -> list[str]:  # including fractions in coefficients
    """
    :return: list of regexes for polynom
    """
    return [
        r"([-+])?(\d*\.\d+|[-+]?\d+(/\d+)?)?x\^([-+]?\d*\.\d+|[-+]?\d+)",
        r"(-?\d+\.\d+|-?\d+)x",
        r"(-?\d+\.\d+|-?\d+)",
    ]


def validate_args() -> tuple[str, bool] | tuple[None, bool]:
    # Validate args using parser
    parser = argparse.ArgumentParser()
    parser.add_argument("equation", nargs=1, type=str, help="Input polynomial equation")
    parser.add_argument(
        "-d",
        "--details",
        type=bool,
        help="Print details",
        default=False,
        const=True,
        nargs="?",
    )

    args = parser.parse_args()
    equation_argument = args.equation[0]
    details = args.details

    # Check if the equation is a string
    if not isinstance(equation_argument, str):
        parser.error("Invalid argument type. Please provide a valid string equation.")
        return None, False

    # Check if there is only one '=' sign
    if equation_argument.count("=") != 1:
        parser.error("Invalid equation format. There should be exactly one '=' sign.")
        return None, False

    return equation_argument, details


def split_equation(equation_string: str) -> tuple[str, str] | None:
    """
    :type equation_string: string
    """

    if not equation_string:
        print("Empty equation")
        return None
    equation_string = equation_string.replace(" ", "").lower()
    equation_string = (
        equation_string.replace("--", "+")
        .replace("-+", "-")
        .replace("+-", "-")
        .replace("-", "+-")
    )

    # Find all occurrences of the pattern and replace them by removing '*'
    equation_string = re.sub(r"(\d+(\\.\d+)?)\*x", r"\1x", equation_string)

    parts = equation_string.split("=")
    equation_left_side = parts[0]
    equation_right_side = parts[1]
    return equation_left_side, equation_right_side


def reformulate_equation_part(equation_part: str) -> dict | None:
    """
    :type equation_part: string
    :param equation_part: one side of the equation string
    :return: dictionary of terms of the equation side
    """
    f_terms = {0: 0}
    terms = equation_part.split("+")
    if "" in terms:
        terms.remove("")
    terms_are_valid = validate_terms(terms)
    if not terms_are_valid:
        return None
    for term in terms:
        if term.find("x") != -1:
            if term.find("^") == -1:
                term += "^1"
            if term.find("x^") == -1:
                return None
            coefficients, degree = term.split("x^")
            if coefficients == "":
                coefficients = "1"
            if coefficients == "-":
                coefficients = "-1"
            if "/" in coefficients:
                numerator, denominator = coefficients.split("/")
                coefficients = float(numerator) / float(denominator)
            try:
                int(degree)
            except ValueError:
                print("Degree", degree, "is not positive integer.")
                return None
            try:
                float(coefficients)
            except ValueError:
                print("Coefficient", coefficients, "is not a valid decimal.")
                return None
            if int(degree) in f_terms.keys():
                f_terms[int(degree)] += float(coefficients)
            else:
                f_terms[int(degree)] = float(coefficients)
        else:
            if 0 in f_terms:
                f_terms[0] += float(term)
            else:
                f_terms[0] = float(term)
    return f_terms


def validate_terms(terms: list[str]) -> bool:
    """
    Validate the terms of the equation
    """
    # Check if the terms syntax are valid
    regx = get_polynom_regex()
    for term in terms:
        if (
            re.fullmatch(regx[0], term) is None
            and re.fullmatch(regx[1], term) is None
            and re.fullmatch(regx[2], term) is None
        ):
            return False
    if terms != []:
        return True
    return False


def reduce(left: dict, right: dict) -> dict:
    """
    Reduce the equation
    """
    # Sum the values of the same  keys and print final dictionary with unique keys
    reduced_terms = left
    reduced_equation = ""
    for degree in right:
        if degree in reduced_terms:
            reduced_terms[degree] -= right[degree]
        else:
            reduced_terms[degree] = -right[degree]
    # sort f_terms by degree from the highest to the lowest
    reduced_terms = dict(sorted(reduced_terms.items(), reverse=True))
    reduced_terms = {
        degree: coefficient
        for degree, coefficient in reduced_terms.items()
        if coefficient != 0
    }
    if not reduced_terms:
        return {0: 0}
    for degree in reduced_terms:
        coefficient = reduced_terms[degree]
        coefficient_str = (
            "{:.0f}".format(coefficient)
            if coefficient.is_integer()
            else str(coefficient)
        )
        if coefficient == 0:
            continue
        if degree == 0:
            if coefficient != 0:
                reduced_equation += coefficient_str
        elif degree == 1:
            if coefficient == 1:
                reduced_equation += "X"
            elif coefficient == -1:
                reduced_equation += "-X"
            else:
                reduced_equation += coefficient_str + " * " + "X"
            reduced_equation += " + " if len(reduced_terms) > 1 else ""
        else:
            if coefficient == 1:
                reduced_equation += "X ^ " + str(degree)
            elif coefficient == -1:
                reduced_equation += "-X ^ " + str(degree)
            else:
                reduced_equation += (
                    (coefficient_str + " * " if coefficient != 1 else "")
                    + "X ^ "
                    + str(degree)
                )
            reduced_equation += " + " if len(reduced_terms) > 1 else ""
    reduced_equation += " = 0"
    reduced_equation = reduced_equation.replace(" +  = 0", " = 0").replace(
        " + -", " - "
    )
    print("Reduced form: ", reduced_equation)
    return reduced_terms


def print_with_details(print_details: bool, *kwargs) -> None:
    """
    Print details
    :param print_details: boolean value if the details should be printed
    :param kwargs: dictionary of arguments
    """
    if print_details:
        print(*kwargs, sep="", end="")


def parentheses(number: float) -> str:
    """
    :param number: number to be printed
    :return: string of the number surrounded by parentheses
    """
    if number < 0:
        return "(" + str(number) + ")"
    else:
        return str(number)


def print_imaginary_number(imaginary_number: complex) -> None:
    """
    Print imaginary number
    :param imaginary_number: imaginary number
    """
    real_part = (
        "{:.6f}".format(imaginary_number.real).rstrip("0").rstrip(".")
        if imaginary_number.real != 0
        else ""
    )
    imaginary_part = (
        "{:+.6f}i".format(imaginary_number.imag).rstrip("0").rstrip(".")
        if imaginary_number.imag != 0
        else ""
    )
    formatted_result = "{}{}".format(
        real_part if real_part != "0" and real_part != "-0" else "",
        imaginary_part if imaginary_part != "0i" and imaginary_part != "-0i" else "",
    )
    if formatted_result == "":
        formatted_result = "0"
    print(formatted_result)


def print_float(number: float) -> None:
    print("{:.6f}".format(number).rstrip("0").rstrip(".") if number != 0 else "0")


def solve_degree_2(details: bool, equation_terms: dict) -> bool:
    """
    Solve the equation of degree 2
    :param details: boolean value if the details should be printed during the solving process
    :param equation_terms: dictionary of terms of the equation
    :return: boolean value if the equation has a solution
    """
    a = equation_terms[2] if 2 in equation_terms else 0
    b = equation_terms[1] if 1 in equation_terms else 0
    c = equation_terms[0] if 0 in equation_terms else 0
    discriminant = b**2 - 4 * a * c
    print_with_details(details, "a = ", a, " | ", "b = ", b, " | ", "c = ", c, "\n")
    print_with_details(
        details,
        "Δ = ",
        "b² - 4ac = ",
        parentheses(b),
        "² - 4 * ",
        parentheses(a),
        " * ",
        parentheses(c),
        " = ",
        discriminant,
        "\n",
    )
    if discriminant < 0:
        print("Discriminant is strictly negative, the two solutions are:")
        print_with_details(
            details,
            "z1 = (-b - i√|Δ|) / 2a = (-",
            parentheses(b),
            " - i√|",
            discriminant,
            "|) / ",
            2 * a,
            " = ",
        )
        res_1 = (-b - discriminant**0.5) / (2 * a)
        print_imaginary_number(res_1)
        print_with_details(
            details,
            "z2 = (-b + i√|Δ|) / 2a = (-",
            parentheses(b),
            " + i√|",
            discriminant,
            "|) / ",
            2 * a,
            " = ",
        )
        res_2 = (-b + discriminant**0.5) / (2 * a)
        print_imaginary_number(res_2)
    elif discriminant == 0:
        print("Discriminant is zero, the solution is:")
        print_with_details(
            details,
            "x = (-b - √|Δ|) / 2a = (-",
            parentheses(b),
            " - √|",
            discriminant,
            "|) / ",
            2 * a,
            " = ",
        )
        res = (-b - discriminant**0.5) / (2 * a)
        print_float(res)
    else:
        print("Discriminant is strictly positive, the two solutions are:")
        print_with_details(
            details,
            "x1 = (-b - √|Δ|) / 2a = (-",
            parentheses(b),
            " - √|",
            discriminant,
            "|) / ",
            2 * a,
            " = ",
        )
        res_1 = (-b - discriminant**0.5) / (2 * a)
        print_float(res_1)
        print_with_details(
            details,
            "x2 = (-b + √|Δ|) / 2a = (-",
            parentheses(b),
            " + √|",
            discriminant,
            "|) / ",
            2 * a,
            " = ",
        )
        res_2 = (-b + discriminant**0.5) / (2 * a)
        print_float(res_2)
    return True


def solve_degree_1(details: bool, equation_terms: dict) -> bool:
    """
    Solve the equation of degree 1
    :param details: boolean value if the details should be printed during the solving process
    :param equation_terms: dictionary of terms of the equation
    :return: boolean value if the equation has a solution
    """
    a = equation_terms[1] if 1 in equation_terms else 0
    b = equation_terms[0] if 0 in equation_terms else 0
    print_with_details(details, "a = ", a, " | ", "b = ", b, "\n")
    print("The solution is:")
    print_with_details(
        details, "x = -b / a = -", parentheses(b), " / ", parentheses(a), " = "
    )
    res = -b / a
    print_float(res)
    return True


def solve_degree_0(equation_terms: dict) -> bool:
    """
    Solve the equation of degree 0
    :param equation_terms: dictionary of terms of the equation
    :return: boolean value if the equation has a solution
    """
    c = equation_terms[0] if 0 in equation_terms else 0
    if c == 0:
        print("All real numbers are solution.")
        return True
    else:
        print("There is no solution.")
        return False


def solve(print_detailed_steps: bool, equation_terms: dict) -> bool:
    """
    Solve the equation
    :param print_detailed_steps: boolean value if the details should be printed during the solving process
    :param equation_terms: dictionary of terms of the equation
    :return: boolean value if the equation has a solution
    """
    # get max degree of the equation terms
    max_degree = max(equation_terms.keys())
    print("Polynomial degree: ", max_degree)
    if max_degree > 2:
        print("The polynomial degree is strictly greater than 2, I can't solve.")
        return False
    elif max_degree == 2:
        return solve_degree_2(print_detailed_steps, equation_terms)
    elif max_degree == 1:
        return solve_degree_1(print_detailed_steps, equation_terms)
    else:
        return solve_degree_0(equation_terms)


if __name__ == "__main__":
    """
    steps:
            1. Validate args
            2. Validate equation
            3. Parse equation
            4. Reduce equation
            5. Solve equation
    """
    equation, print_details = validate_args()
    left_equation_part, right_equation_part = split_equation(equation)
    left_terms = reformulate_equation_part(left_equation_part)
    right_terms = reformulate_equation_part(right_equation_part)
    if left_terms is None or right_terms is None:
        print("Invalid equation syntax.")
        exit(1)
    equation_dict = reduce(left_terms, right_terms)
    solved = solve(print_details, equation_dict)
    if solved is None:
        print("The polynomial degree is strictly greater than 2, I can't solve.")
        exit(1)
