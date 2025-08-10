<script>
	import '@fortawesome/fontawesome-free/css/all.min.css';
	let { question, url, question_date, questioner, politician, subject_area, answer, answer_date } = $props();
	
</script>

<h2>Frage {#if questioner !== null}von {questioner}{/if} an {politician}</h2>
<a href={url} target="_blank" rel="noopener noreferrer">
	<i class="fa-solid fa-link"></i>
</a>

<div>
	<div>Frage</div>
	<div>{question_date}</div>
	<p>{question}</p>
	{#if subject_area !== null}<div>Thema: {subject_area}</div>{/if}
</div>
<div>
	{#if answer !== null}
		<div>Antwort</div>
		<div>{answer_date}</div>
		<p>{answer}</p>
	{:else}
		<p>Keine Antwort</p>
	{/if}
</div>