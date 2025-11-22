{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes %}
Attributes table
~~~~~~~~~~~~~~~~

.. autosummary::
{% for item in attributes %}
    {%- if item not in inherited_members and item not in ['training'] %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
Methods table
~~~~~~~~~~~~~

.. autosummary::
{% for item in methods %}
    {%- if item != '__init__' and item not in inherited_members %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
Attributes
~~~~~~~~~~

{% for item in attributes %}
    {%- if item not in inherited_members and item not in ['training'] %}

.. autoattribute:: {{ [objname, item] | join(".") }}
    {%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods %}
Methods
~~~~~~~

{% for item in methods %}
    {%- if item != '__init__' and item not in inherited_members %}


.. automethod:: {{ [objname, item] | join(".") }}
    {%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
