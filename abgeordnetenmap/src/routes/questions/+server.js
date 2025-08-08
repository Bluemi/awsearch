// src/routes/api/file/[name]/+server.ts
import { error } from '@sveltejs/kit';
import fs from 'node:fs/promises';
import path from 'node:path';

let cache = null;

async function loadFile() {
	const filePath = path.resolve(`./data/complete_export.json`);

	if (cache === null) {
		cache = await fs.readFile(filePath);
		cache = JSON.parse(cache.toString());
	}

	return cache;
}

export async function POST({ request }) {
	// console.log('request:', request);
	const ids = await request.json();
	try {
		const questionBase = await loadFile();
		const questions = ids.map(id => questionBase.questions[id]);
		return new Response(JSON.stringify(questions), {
			headers: {
				'Content-Type': 'application/json',
			}
		});
	} catch (e) {
		if (e.status && e.body) throw e;
		console.error(e);
		throw error(500, 'Error loading file: ' + (e?.message ?? e));
	}
}
