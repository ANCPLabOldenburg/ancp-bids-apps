import sys


class App:
    def get_args_parser(self):
        raise NotImplemented

    def execute(self, **kwargs):
        raise NotImplemented


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--app', type=str, required=True, help="the (registered) app to run, example: "
                                                                     "-a nilearnfirstlevelapp")
    parser.add_argument('-l', '--list', action='store_true', help="a list of all registered apps")
    parser.add_argument('--args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ATM only internal apps supported
    # TODO allow dynamic loading of external apps
    from ancpbidsapps import APPS

    if args.list:
        for app in APPS.keys():
            print("Key: %s\nDescription: %s\n---\n" % (app, APPS[app].__doc__.strip()))
    else:
        app = APPS[args.app.lower()]()
        app_args_parser = app.get_args_parser()
        app_args = vars(app_args_parser.parse_args(args.args))
        app.execute(**app_args)
        print(args)
