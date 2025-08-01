<script>
</script>

<style>
    nav {
        background-color: #f8f9fa;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .nav-brand {
        font-size: 1.25rem;
        font-weight: bold;
        color: #333;
        text-decoration: none;
    }

    .search-section {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 45px;
        flex-wrap: wrap;
				padding: 2rem;
    }

    .search-header {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .search-title {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 1rem;
    }

    .search-title-small {
        font-size: 1.2rem;
        color: #666;
				padding-left: 10px;
    }

    .search-title-large {
        font-size: 1.5rem;
        font-weight: bold;
    }

    .search-input {
        flex-grow: 1;
        max-width: 250px;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
</style>

<nav>
	<a href="/" class="nav-brand">AbgeordnetenMap</a>
</nav>

<section class="search-section">
	<div class="search-header">
		<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
				 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<circle cx="11" cy="11" r="8"></circle>
			<line x1="21" y1="21" x2="16.65" y2="16.65"></line>
		</svg>
		<div class="search-title">
			<span class="search-title-small">Finde deine</span>
			<span class="search-title-large">Fragen und Antworten</span>
		</div>
	</div>
	<input type="text" class="search-input" placeholder="Suche Thema">
</section>

