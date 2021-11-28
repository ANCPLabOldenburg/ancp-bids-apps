class App:
    def get_args_parser(self):
        raise NotImplemented

    def execute(self, **kwargs):
        raise NotImplemented

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--app', type=str, help="the app to run")
    parser.add_argument('--args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ATM only internal apps supported
    # TODO allow dynamic loading of external apps
    from ancpbidsapps import APPS
    app = APPS[args.app]()
    app_args_parser = app.get_args_parser()
    app_args = vars(app_args_parser.parse_args(args.args))
    app.execute(**app_args)
    print(args)

