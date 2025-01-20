---
permalink: /blog/
title: "Entropy Blog"
seo_title: "Alessio Devoto's blog"
layout: single
author_profile: true
---


{{ content }}

{% assign entries_layout = page.entries_layout | default: 'list' %}
<div class="entries-{{ entries_layout }}">
  {% for post in site.posts %}
    {% include archive-single.html type=entries_layout %}
    <p class="page__meta">
      <i class="far fa-calendar-alt" aria-hidden="true"></i> 
      {{ post.date | date: "%B %d, %Y" }}
    </p>
  {% endfor %}
</div>


