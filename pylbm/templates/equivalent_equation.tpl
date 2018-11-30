{%- from 'wrap.tpl' import wrap -%}
{{ header }}
    The equation is

    {{ wrap(phys_equation) | indent(4, True)}}

    where

    {{ wrap(conserved_moments) | indent(4, True)}}

    {%- for o in order1 %}
    {{ wrap(o) | indent(4, True)}}
    {%- endfor %}

    {%- for o in order2 %}
    {{ wrap(o) | indent(4, True)}}
    {%- endfor %}
