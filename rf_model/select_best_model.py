import click
from typing import TypeAlias
from operator import itemgetter

Model: TypeAlias = tuple[str, int, float]


def read_error_rate(file_path: str) -> Model:
    with open(file_path) as f_input:
        f_input.readline()
        match f_input.readline().strip().split(sep="\t"):
            case [num_feature, error_rate]:
                return (file_path, int(num_feature), float(error_rate))
            case _:
                raise Exception(file_path)


def filter_by_error_rate(error_rates: list[Model]) -> list[Model]:
    _,_,min_error_rate = min(error_rates, key=itemgetter(2))
    return [k for k in error_rates if k[2] == min_error_rate]


def filter_by_num_features(error_rates: list[Model]) -> list[Model]:
    _,min_num_features,_ = min(error_rates, key=itemgetter(1))
    return [k for k in error_rates if k[1] == min_num_features]


@click.command()
@click.argument("error_rate_files", nargs=-1)
def select_best_model(error_rate_files):
    error_rate_list = [read_error_rate(k) for k in error_rate_files]
    print((filter_by_num_features(filter_by_error_rate(error_rate_list)))[0][0])


if __name__ == "__main__":
    select_best_model()
