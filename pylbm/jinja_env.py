from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader('pylbm', 'templates'),
    autoescape=select_autoescape(['tpl'])
)