import type { APIRoute } from 'astro';
import testData from '../../../backend/testdata.json';

export const GET: APIRoute = async () => {
  // Transform the data into the format expected by the scatter plot
  const scatterData = testData.map((point, index) => ({
    x: point[0],
    y: point[1],
    id: index
  }));

  return new Response(JSON.stringify(scatterData), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  });
};