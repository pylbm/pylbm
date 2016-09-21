{%- extends 'rst.tpl' -%}

{% block stream %}
.. parsed-literal::
    :class: output

{{ output.text | indent }}
{% endblock stream %}

{% block data_text scoped %}
.. parsed-literal::
    :class: output
    
{{ output.data['text/plain'] | indent }}
{% endblock data_text %}