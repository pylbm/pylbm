{{ header }}
    - dimension: {{ elem.dim }}
    - filename: {{ elem.filename }}
    - number of triangles: {{ elem.nb_tri }}
    - label: {{ elem.label }}
    - type: {{ type }}
    - spatial extent:
        . x: {{ dim[0] }}, {{ dim[3] }}
        . y: {{ dim[1] }}, {{ dim[4] }}
        . z: {{ dim[2] }}, {{ dim[5] }}
