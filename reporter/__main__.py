from reporter import components
from reporter.publish import publish


def main():
    comps = [
        components.header(),
        components.config_diff(),
        components.general(),
        components.footer(),
    ]
    report = "".join(comps)

    print(report)
    import os

    publish(report)


if __name__ == "__main__":
    main()
