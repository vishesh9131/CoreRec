"""Serving stack smoke tests (requires corerec[serving])."""
import unittest


class TestServingSmoke(unittest.TestCase):
    def test_fastapi_import_and_server(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("Install corerec[serving] for serving tests")

        from corerec.serving.model_server import FASTAPI_AVAILABLE, ModelServer

        self.assertTrue(FASTAPI_AVAILABLE)

        from corerec.engines.collaborative import SAR
        import pandas as pd

        df = pd.DataFrame(
            {"userID": [0, 0, 1, 1], "itemID": [10, 11, 10, 12], "rating": [5, 4, 3, 5]}
        )
        model = SAR()
        model.fit(df)

        server = ModelServer(model)
        client = TestClient(server.app)

        resp = client.post("/recommend", json={"user_id": 0, "top_k": 2})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("recommendations", body)
        self.assertGreaterEqual(len(body["recommendations"]), 1)


if __name__ == "__main__":
    unittest.main()
