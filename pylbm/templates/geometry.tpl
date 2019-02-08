{%- from 'wrap.tpl' import wrap -%}
{{ header }}
    - spatial dimension: {{ geom.dim }}
    - bounds of the box: {{ geom.bounds|join(' x ') }}
    - labels: {{ geom.box_label }}
    {%- if geom.list_elem %}
    - list of elements added or deleted in the box
        {%- for elem in geom.list_elem -%}
        {{ wrap(elem)| indent(8, True) }}
        {%- endfor %}
    {%- endif %}