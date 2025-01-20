---
permalink: /archive/
layout: archive
---


{% assign entries_layout = page.entries_layout | default: 'list' %}
<div class="entries-{{ entries_layout }}">
  {% for post in site.posts %}
    {% unless post.hidden %}
      <div class="archive__item">
        <h2 class="archive__item-title">
          <a href="{{ post.url }}">{{ post.title }}</a>
        </h2>
        {% assign words = post.content | number_of_words %}
        {% assign reading_time = words | divided_by: 180 | plus: 1 %}
        <p class="archive__item-meta">
          {{ post.date | date: "%B %d, %Y" }} • {{ reading_time }} min read
          {% if post.excerpt %}
            <br>{{ post.excerpt | markdownify | strip_html | truncate: 160 }}
          {% endif %}
        </p>
      </div>
    {% endunless %}
  {% endfor %}
</div>