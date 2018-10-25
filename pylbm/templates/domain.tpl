{%- from 'wrap.tpl' import wrap -%}
{{ header }}
    - spatial dimension: {{ dom.dim }}
    - space step: {{ dom.dx }}
    - with halo:
        bounds of the box: {{ dom.get_bounds_halo()|join(' x ') }}
        number of points: {{ dom.shape_halo }}
    - without halo:
        bounds of the box: {{ dom.get_bounds()|join(' x ') }}
        number of points: {{ dom.shape_in }}
    {{ wrap(dom.geom) | indent(4, True) }}
