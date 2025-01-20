---
permalink: /archive/
layout: archive
---


{% assign entries_layout = page.entries_layout | default: 'list' %}
<div class="entries-{{ entries_layout }}">
  {% for post in site.posts %}
    {% unless post.hidden %}
      {% include archive-single.html type=entries_layout %}
    {% endunless %}
  {% endfor %}
</div>