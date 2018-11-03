{%- from 'wrap.tpl' import wrap -%}
{{ header }}
    - spatial dimension: {{ scheme.dim }}
    - number of schemes: {{ scheme.nschemes }}
    - number of velocities: {{ scheme.stencil.nv[-1] }}
    - conserved moments: {{ consm }}
    {%- for k in range(scheme.nschemes) %}
    {{ wrap(header_scheme[k]) | indent(4, True)}}
        - velocities
        {%- for v in scheme.stencil.v[k] %}
            {{ v }}
        {%- endfor %}

        - polynomials
        {{ wrap(P[k]) | indent(12, True) }}

        - equilibria
        {{ wrap(EQ[k]) | indent(12, True) }}

        - relaxation parameters
        {{ wrap(s[k]) | indent(12, True) }}

    {%- endfor %}

    - moments matrices
        {{ wrap(M) | indent(8, True) }}

    - inverse of moments matrices
        {{ wrap(invM) | indent(8, True) }}

    {% if rel_vel is defined %}
    - relative velocities
        {{ wrap(rel_vel) | indent(8, True) }}

    - Transition matrix
        {{ wrap(Tu) | indent(8, True) }}
    {%- endif %}
