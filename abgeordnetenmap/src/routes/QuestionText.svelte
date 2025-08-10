<script>
	import '@fortawesome/fontawesome-free/css/all.min.css';
	let { question, url, question_date, questioner, politician, subject_area, answer, answer_date } = $props();
	
</script>

<h2>Frage {#if questioner !== null}von {questioner}{/if} an {politician}</h2>
<a href={url} target="_blank" rel="noopener noreferrer" aria-label="Link zur Frage">
	<i class="fa-solid fa-link"></i>
</a>

<div class="question-container">
	<div class="section-header">Frage</div>
	<div class="date">{question_date}</div>
	<p class="text">{question}</p>
	{#if subject_area !== null}
		<div class="subject">Thema: {subject_area}</div>
	{/if}
</div>
<div class="answer-container">
	{#if answer !== null}
		<div class="section-header">Antwort</div>
		<div class="date">{answer_date}</div>
		<p class="text">{answer}</p>
	{:else}
		<p class="no-answer">Keine Antwort</p>
	{/if}
</div>

<style>
    .question-container, .answer-container {
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f5f5f5;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .date {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1rem;
    }

    .text {
        line-height: 1.6;
        margin: 0;
        color: #2c2c2c;
    }

    .subject {
        margin-top: 1rem;
        font-style: italic;
        color: #555;
    }

    .no-answer {
        color: #888;
        font-style: italic;
        margin: 0;
    }

    .answer-container {
        background-color: #e8f4f8;
    }
</style>