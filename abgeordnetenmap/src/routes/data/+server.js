import { error } from '@sveltejs/kit';
import fs from 'fs/promises';

export async function GET() {
	const filePath = './data/preview_export.bin';
	try {
		const fileData = await fs.readFile(filePath);

		return new Response(fileData, {
			headers: {
				'Content-Type': 'application/octet-stream',
				'Content-Disposition': 'attachment; filename="preview.bin"',
			},
		});
	} catch (e) {
		throw error(500, 'Error reading file: ' + e.message);
	}
}
