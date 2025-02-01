---
permalink: /blog/
title: "Entropy Blog"
seo_title: "Alessio Devoto's blog"
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
        <span class="post-meta">
          {{ post.date | date: "%B %d, %Y" }} • {{ reading_time }} min read
        </span>
        {% if post.excerpt %}
          <p class="archive__item-excerpt">
            {{ post.excerpt | markdownify | strip_html | truncate: 160 }}
          </p>
        {% endif %}
      </div>
    {% endunless %}
  {% endfor %}
</div>

<style>
.post-meta {
  display: inline-block;
  padding: 4px 8px;
  background-color: #f2f2f2;
  border-radius: 4px;
  font-size: 0.85em;
  color: #666;
}
</style>
