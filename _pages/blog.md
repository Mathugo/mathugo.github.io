---
permalink: /blog/
title: "Entropy Blog"
seo_title: "Alessio Devoto's blog"
layout: single
author_profile: true
---

{% for post in site.posts %}
  {% assign words = post.content | strip_html | number_of_words %}
  <div class="archive__item">
    <h2 class="archive__item-title no_toc" itemprop="headline">
      <a href="{{ post.url | relative_url }}" rel="permalink">{{ post.title }}</a>
    </h2>
    {% if post.excerpt %}<p class="archive__item-excerpt" itemprop="description">{{ post.excerpt | markdownify | strip_html | truncate: 160 }}</p>{% endif %}
    <p class="page__meta">
      <i class="far fa-clock" aria-hidden="true"></i> 
      {% assign words_per_minute = site.words_per_minute | default: 200 %}
      {% if words < words_per_minute %}
        ~1 min read
      {% else %}
        {{ words | divided_by:words_per_minute | round }} min read
      {% endif %}
    </p>
  </div>
{% endfor %}

