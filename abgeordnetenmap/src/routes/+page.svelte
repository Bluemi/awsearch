<script lang="ts">
	import { onMount } from 'svelte';
	import { abgeordnetenmap } from '$lib/questionbase/proto-bundle';
	import { MaxPlot } from "$lib/maxPlot"

	let questions = $state(['Some question']);
	let blob = $state(new ArrayBuffer(0));

	async function loadPreview() {
		const response = await fetch('/data/preview_export.bin');
		const data = await response.arrayBuffer();
		let result = await decodeArrayBuffer(data);
		if (result !== null) {
			createPlot(result);
		}
		blob = data;
	}

	function createPlot(preview: abgeordnetenmap.PreviewQuestionBase) {
		const div = document.getElementById("map");

		if (div === null) {
			return
		}
		
		let fig = new MaxPlot(div, 50, 50, div.clientWidth, div.clientHeight, {'radius':30, 'alpha':1.0});

		fig.canvas.style.border = "1px solid black";

		function initializePlot(fig) {
		// draw five little circles
			fig.initPlot({'radius':3, 'alpha':0.8});
			const coords = [];
			const colors = [];
			for (let q of preview.questions) {
				coords.push(q.x);
				coords.push(q.y);
				colors.push(q.clusterId === undefined?0:q.clusterId);
			}
			fig.setCoords(
				coords,
				[],
				{}, {}
			);
			// set different colors
			fig.setColors([
				"ff0000", "ff8000", "ffff00", "80ff00",
				"00ff00", "00ff80", "00ffff", "0080ff",
				"0000ff", "8000ff", "ff00ff", "ff0080",
				"808080", "ffc000", "404040", "ff6666",
				"2020f6", "6666ff", "ffa366", "66aaa3",
				"a366ff", "ffff99", "99ffff", "ff99ff",
				"bdaa2e", "9999ff", "ccbbaa", "aabbcc",
				"bbaacc", "996633", "339966", "663399",
				"993366", "669933", "336699"
			]);
			// and assign every circle its own color
			fig.setColorArr(colors);
			// and draw it
			fig.drawDots();
		}

		initializePlot(fig);

		fig.onSelChange = function(cellIds) {
			// console.log('selected:', cellIds.length);
		};
		fig.onCellHover = function(cellIds) {
			if (cellIds===null) {
				// console.log('nothing hovered');
			} else {
				// console.log('hovered:', cellIds.length);
			}
		};
		fig.onCellClick = function(cellIds) {
			if (cellIds===null) {
				// console.log('nothing clicked');
			} else {
				// console.log('clicked:', cellIds.length);
			}
		};
		fig.onNoLabelHover = function(ev) {
			// console.log('no label hovered');
		}
	}

	async function decodeArrayBuffer(arrayBuffer: ArrayBuffer): abgeordnetenmap.PreviewQuestionBase | null {
		try {
			const buffer = new Uint8Array(arrayBuffer);

			const message = abgeordnetenmap.PreviewQuestionBase.decode(buffer);

			return abgeordnetenmap.PreviewQuestionBase.toObject(message);
		} catch (e) {
			let error = '';
			if (e instanceof Error) {
				error = `Error decoding Protobuf message: ${e.message}`;
			} else {
				error = `An unknown error occurred during decoding.`;
			}
			console.error(error);
		}
		return null;
	}

	onMount(() => {
		loadPreview();
	});
</script>

<style>
    .container {
        padding: 10px;
        margin: auto;
        width: 85%;
        min-height: 100vh;
        background-color: #dfe2e3;
        @media (max-width: 768px) {
            width: 100%;
            margin: 0;
        }
    }

    nav {
        background-color: #e9ecef;
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
        gap: 5%;
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
        font-size: 1.0rem;
        flex-grow: 1;
        max-width: 250px;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }

		.map-section {
			display: flex;
			flex-direction: row;
			justify-content: flex-start;
			gap: 2.5rem;
			padding: 1rem;
		}

		#map {
				height: 60vh;
				width: 60%;
				background-color: #babdbe;
		}

		.question-section {
				width: 45%;
		}

		.question-header {
				margin-top: 0;
		}
</style>

<main>
	<nav>
		<a href="/" class="nav-brand">AbgeordnetenMap</a>
	</nav>
	<div class="container">

		<section class="search-section">
			<div class="search-header">
				<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor"
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

		<section class="map-section">
			<div id="map"></div>
			<div class="question-section">
				<h1 class="question-header">Fragen</h1>
				loaded {blob.byteLength} bytes
				{#if questions.length === 0}
					Keine Fragen ausgewählt
				{:else}
					{#each questions as question}
						<div class="question">{question}</div>
					{/each}
				{/if}
			</div>
		</section>

	</div>
</main>

