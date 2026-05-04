declare module 'threejs-components/build/cursors/tubes1.min.js' {
  export type TubesCursorApp = {
    tubes: {
      setColors: (colors: string[]) => void;
      setLightsColors: (colors: string[]) => void;
    };
  };
  function TubesCursor(
    canvas: HTMLCanvasElement,
    options: {
      tubes: {
        colors: string[];
        lights: { intensity: number; colors: string[] };
      };
    }
  ): TubesCursorApp;
  export default TubesCursor;
}
