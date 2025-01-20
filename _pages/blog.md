---
permalink: /blog/
title: "Entropy Blog"
seo_title: "Alessio Devoto's blog"
layout: single
author_profile: true
---

{% for post in site.posts %}
  {% assign words = post.content | strip_html | number_of_words %}
  {% include archive-single.html %}
  <p class="page__meta">
    <i class="far fa-clock" aria-hidden="true"></i> 
    {% assign words_per_minute = site.words_per_minute | default: 200 %}
    {% if words < words_per_minute %}
      ~1 min read
    {% else %}
      {{ words | divided_by:words_per_minute | round }} min read
    {% endif %}
  </p>
{% endfor %}

