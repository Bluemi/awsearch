import { error } from '@sveltejs/kit';
import fs from 'fs/promises';

export async function GET() {
	const filePath = './data/preview_export.bin';  // Replace with the actual path to your file
	try {
		const fileData = await fs.readFile(filePath);

		return new Response(fileData, {
			headers: {
				'Content-Type': 'application/octet-stream',
				'Content-Disposition': 'attachment; filename="yourfile.bin"',
			},
		});
	} catch (e) {
		throw error(404, 'Error reading file: ' + e.message);
	}
}
