{{ header }}
    - spatial dimension: {{ stencil.dim }}
    - minimal velocity in each direction: {{ stencil.vmin }}
    - maximal velocity in each direction: {{ stencil.vmax }}
    - information for each elementary stencil:
    {%- for k in range(stencil.nstencils) %}
        stencil {{ k }}
            - number of velocities: {{ stencil.nv[k] }}
            - velocities
            {%- for v in stencil.v[k] %}
                {{ v }}
            {%- endfor %}
    {%- endfor %}