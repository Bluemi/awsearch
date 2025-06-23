window.addEventListener('load', () => {
  const container = document.getElementById('plotContainer');
  const figDiv = document.createElement('div');
  container.appendChild(figDiv);
  const fig = new MaxPlot(figDiv, 50, 50, 800, 400, { radius: 5, alpha: 0.8 });
  fig.canvas.style.border = '1px solid black';

  fig.onSelChange = ids => document.getElementById('selected').value = ids.length + ' selected';
  fig.onCellHover = ids => document.getElementById('hover').value = ids ? ids.length + ' hover' : '';
  fig.onCellClick = ids => document.getElementById('clicked').value = ids ? ids.length + ' clicked' : '';
  fig.onNoLabelHover = () => {};

  document.getElementById('loadData').addEventListener('click', () => {
    fetch('/data')
      .then(res => res.json())
      .then(data => {
        const coords = [];
        data.forEach(p => coords.push(p.x, p.y));
        fig.initPlot({ radius: 5, alpha: 0.8 });
        fig.setCoords(coords, []);
        // set a palette and map each point to a color index
        const palette = ['ff0000', '00ff00', '0000ff', '000000'];
        fig.setColors(palette);
        const mapping = data.map((_, i) => i % palette.length);
        fig.setColorArr(mapping);
        fig.drawDots();
      })
      .catch(err => console.error('fetch error:', err));
  });

  document.getElementById('clear').addEventListener('click', () => {
    fig.selectClear();
    fig.drawDots();
  });
});
