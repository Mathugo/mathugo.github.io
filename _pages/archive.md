---
permalink: /archive/
layout: archive
---



{% assign entries_layout = page.entries_layout | default: 'list' %}
{% assign posts = site.posts | where_exp: "item", "item.hidden != true" %}

<div class="entries-{{ entries_layout }}">
  {% for post in posts %}
    {% assign words = post.content | strip_html | number_of_words %}
    {% assign reading_time = words | divided_by: 180 | plus: 1 %}
    
    <article class="archive__item">
      <h2 class="archive__item-title no_toc" itemprop="headline">
        {% if post.link %}
          <a href="{{ post.link }}">{{ post.title }}</a> <a href="{{ post.url | relative_url }}" rel="permalink"><i class="fas fa-link" aria-hidden="true" title="permalink"></i><span class="sr-only">Permalink</span></a>
        {% else %}
          <a href="{{ post.url | relative_url }}" rel="permalink">{{ post.title }}</a>
        {% endif %}
      </h2>
      
      {% if post.excerpt %}
        <p class="archive__item-excerpt" itemprop="description">
          {{ post.excerpt | markdownify | strip_html | truncate: 160 }}
        </p>
      {% endif %}
      
      <p class="page__meta">
        <span class="page__meta-readtime">
          {% if reading_time <= 1 %}1 minute read
          {% else %}{{ reading_time }} minute read{% endif %}
        </span>
        {% if post.date %}
          <span class="page__meta-date">
            <i class="far fa-calendar-alt" aria-hidden="true"></i>
            {{ post.date | date: "%B %-d" }}
          </span>
        {% endif %}
      </p>
    </article>
  {% endfor %}
</div>
