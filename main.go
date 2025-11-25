package search_vector_go

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	elasticsearch7 "github.com/elastic/go-elasticsearch/v7"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Vector []float32 `json:"vector"`
}

type BatchEmbedRequest struct {
	Texts []string `json:"texts"`
}

type BatchEmbedResponse struct {
	Vectors [][]float32 `json:"vectors"`
	Count   int         `json:"count"`
}

type Server struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
	mu       sync.RWMutex
}

var (
	globalServer *Server
	initOnce     sync.Once
	initError    error
)

// ---------- Types ----------
type TargetCriteria struct {
	Operator string   `json:"operator"`
	Values   []string `json:"values"`
}

type TargetRule struct {
	ID     int                       `json:"id"`
	Target map[string]TargetCriteria `json:"target"`
}

type EmbedReq struct {
	Text string `json:"text"`
}

type EmbedRes struct {
	Vector []float64 `json:"vector"`
}

// ---------- Config ----------
const IndexName = "target_keywords"
const Dim = 384

func initServer() (*Server, error) {
	initOnce.Do(func() {
		log.Println("🔄 Initializing embedding server...")
		start := time.Now()

		// Create pure Go session
		session, err := hugot.NewGoSession()
		if err != nil {
			initError = fmt.Errorf("failed to create session: %w", err)
			return
		}

		// Download options - ensure we get the right ONNX model
		downloadOptions := hugot.NewDownloadOptions()
		downloadOptions.OnnxFilePath = "onnx/model.onnx"

		log.Println("📥 Downloading all-MiniLM-L6-v2 model...")
		modelPath, err := hugot.DownloadModel(
			"sentence-transformers/all-MiniLM-L6-v2",
			"./models",
			downloadOptions,
		)
		if err != nil {
			session.Destroy()
			initError = fmt.Errorf("failed to download model: %w", err)
			return
		}

		// Create pipeline config with explicit settings
		config := hugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         "embeddingPipeline",
			OnnxFilename: "onnx/model.onnx",
		}

		log.Println("🔧 Loading model into pipeline...")
		pipeline, err := hugot.NewPipeline(session, config)
		if err != nil {
			session.Destroy()
			initError = fmt.Errorf("failed to create pipeline: %w", err)
			return
		}

		globalServer = &Server{
			session:  session,
			pipeline: pipeline,
		}

		log.Printf("✅ Server initialized successfully in %v", time.Since(start))

		// Test embedding to verify
		testEmbed, err := globalServer.computeEmbedding("test")
		if err != nil {
			log.Printf("⚠️  Warning: Test embedding failed: %v", err)
		} else {
			log.Printf("✅ Test embedding successful (dim=%d)", len(testEmbed))
		}
	})

	return globalServer, initError
}

func (s *Server) Close() {
	if s.session != nil {
		log.Println("🛑 Shutting down server...")
		s.session.Destroy()
	}
}

func meanPooling(tokenEmbeddings [][]float32, attentionMask []int64) []float32 {
	if len(tokenEmbeddings) == 0 {
		return nil
	}

	seqLen := len(tokenEmbeddings)
	hiddenSize := len(tokenEmbeddings[0])
	result := make([]float32, hiddenSize)

	for i := 0; i < hiddenSize; i++ {
		sum := float32(0)
		count := float32(0)

		for j := 0; j < seqLen; j++ {
			if j < len(attentionMask) && attentionMask[j] == 1 {
				sum += tokenEmbeddings[j][i]
				count += 1
			}
		}

		if count > 0 {
			result[i] = sum / count
		}
	}

	return result
}

// normalizeL2 normalizes vector to L2 norm = 1
func normalizeL2(embedding []float32) []float32 {
	var sumSquares float64
	for _, v := range embedding {
		sumSquares += float64(v) * float64(v)
	}

	if sumSquares == 0 {
		return embedding
	}

	norm := math.Sqrt(sumSquares)
	normalized := make([]float32, len(embedding))
	for i, v := range embedding {
		normalized[i] = float32(float64(v) / norm)
	}

	return normalized
}

func (s *Server) computeEmbedding(text string) ([]float32, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Create single-item batch
	batch := []string{text}

	// Run pipeline
	result, err := s.pipeline.RunPipeline(batch)
	if err != nil {
		return nil, fmt.Errorf("pipeline execution failed: %w", err)
	}

	// Validate result
	if len(result.Embeddings) == 0 || len(result.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("empty embedding generated")
	}

	embedding := result.Embeddings[0]

	// Apply L2 normalization (sentence-transformers does this by default)
	normalized := normalizeL2(embedding)

	return normalized, nil
}

func embedHandler(text string) ([]float32, error) {

	embedding, err := globalServer.computeEmbedding(text)
	if err != nil {
		log.Printf("❌ Embedding error: %v", err)
		return nil, err
	}

	// Calculate sum for verification
	var sum float64
	for _, v := range embedding {
		sum += float64(v)
	}

	return embedding, nil
}

// ---------- Helpers ----------
// func getEmbedding(text string) ([]float32, error) {
// 	reqBody, _ := json.Marshal(map[string]string{"text": text})
// 	resp, err := http.Post(EmbedServer, "application/json", bytes.NewReader(reqBody))
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer resp.Body.Close()

// 	var out EmbedRes
// 	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
// 		return nil, err
// 	}

// 	// convert float64 -> float32
// 	vec32 := make([]float32, len(out.Vector))
// 	for i, v := range out.Vector {
// 		vec32[i] = float32(v)
// 	}
// 	return vec32, nil
// }

// ---------- ES helpers ----------
func createIndex(es *elasticsearch7.Client) error {
	mapping := map[string]interface{}{
		"settings": map[string]interface{}{
			"analysis": map[string]interface{}{
				"analyzer": map[string]interface{}{
					"vietnamese_analyzer": map[string]interface{}{
						"type":      "custom",
						"tokenizer": "standard",
						"filter": []string{
							"lowercase",
							"asciifolding", // Chuyển tiếng Việt có dấu sang không dấu
						},
					},
				},
			},
		},
		"mappings": map[string]interface{}{
			"properties": map[string]interface{}{
				"id":               map[string]interface{}{"type": "integer"},
				"rule_id":          map[string]interface{}{"type": "integer"},
				"original_keyword": map[string]interface{}{"type": "keyword"},
				// Thêm text field để fuzzy search
				"keyword_text": map[string]interface{}{
					"type":     "text",
					"analyzer": "vietnamese_analyzer",
				},
				"keyword_normalized": map[string]interface{}{
					"type": "keyword", // Keyword không dấu để exact match dễ hơn
				},
				"embedding": map[string]interface{}{
					"type": "dense_vector",
					"dims": Dim,
				},
				// Flatten target criteria để có thể query được
				"target_country":  map[string]interface{}{"type": "keyword"},
				"target_device":   map[string]interface{}{"type": "keyword"},
				"target_platform": map[string]interface{}{"type": "keyword"},
				"target_language": map[string]interface{}{"type": "keyword"},
			},
		},
	}

	body, _ := json.Marshal(mapping)
	// Xóa index cũ nếu có
	// es.Indices.Delete([]string{IndexName})
	res, err := es.Indices.Create(IndexName, es.Indices.Create.WithBody(bytes.NewReader(body)))
	if err != nil {
		return err
	}
	defer res.Body.Close()
	return nil
}

func indexKeywordDoc(es *elasticsearch7.Client, rule TargetRule, keyword string, vec []float32) error {
	// Flatten target object thành các field riêng biệt
	doc := map[string]interface{}{
		"rule_id":            rule.ID,
		"id":                 rule.ID,
		"original_keyword":   keyword,
		"keyword_text":       keyword, // Để fuzzy search
		"keyword_normalized": keyword, // Keyword không dấu
		"embedding":          vec,
	}

	// Thêm các target fields
	for key, crit := range rule.Target {
		if key == "keyword" {
			continue // Skip keyword field
		}
		fieldName := fmt.Sprintf("target_%s", key)
		doc[fieldName] = crit.Values
	}

	docJSON, _ := json.Marshal(doc)
	res, err := es.Index(IndexName, bytes.NewReader(docJSON), es.Index.WithRefresh("true"))
	if err != nil {
		return err
	}
	defer res.Body.Close()
	return nil
}

func buildESQuery(qvec []float32, variables map[string]TargetCriteria, minScore float64, query string) map[string]interface{} {
	filterClauses := []interface{}{}

	// Build filter clauses
	for key, crit := range variables {
		// Filter out empty string values
		validValues := []interface{}{}
		for _, v := range crit.Values {
			// Convert to string and check if not empty
			if strVal := fmt.Sprint(v); strVal != "" {
				validValues = append(validValues, v)
			}
		}

		// Skip if no valid values
		if len(validValues) == 0 {
			continue
		}

		fieldName := fmt.Sprintf("target_%s", key)

		if crit.Operator == "include" {
			filterClauses = append(filterClauses, map[string]interface{}{
				"terms": map[string]interface{}{
					fieldName: validValues,
				},
			})
		} else if crit.Operator == "exclude" {
			filterClauses = append(filterClauses, map[string]interface{}{
				"bool": map[string]interface{}{
					"must_not": []interface{}{
						map[string]interface{}{
							"terms": map[string]interface{}{
								fieldName: validValues,
							},
						},
					},
				},
			})
		}
	}

	normalizedQuery := query

	// Determine the base query based on whether we have filters
	var baseQuery map[string]interface{}

	if len(filterClauses) > 0 {
		// Use bool query with filters when we have filter criteria
		baseQuery = map[string]interface{}{
			"bool": map[string]interface{}{
				"filter": filterClauses,
				"should": []interface{}{
					// Boost if fuzzy text matches
					map[string]interface{}{
						"match": map[string]interface{}{
							"keyword_text": map[string]interface{}{
								"query":     query,
								"fuzziness": "AUTO",
								"boost":     0.5,
							},
						},
					},
					// Boost if normalized matches
					map[string]interface{}{
						"term": map[string]interface{}{
							"keyword_normalized": map[string]interface{}{
								"value": normalizedQuery,
								"boost": 0.5,
							},
						},
					},
				},
				"minimum_should_match": 0,
			},
		}
	} else {
		// Use match_all when no filters are specified
		baseQuery = map[string]interface{}{
			"match_all": map[string]interface{}{},
		}
	}

	// Build the final query with script_score
	esQuery := map[string]interface{}{
		"query": map[string]interface{}{
			"script_score": map[string]interface{}{
				"query": baseQuery,
				"script": map[string]interface{}{
					"source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
					"params": map[string]interface{}{
						"query_vector": qvec,
					},
				},
			},
		},
	}

	// Only apply min_score if specified and > 0
	if minScore > 0 {
		esQuery["min_score"] = minScore
	}

	return esQuery
}

// Helper functions for log formatting
func findVectorInJSON(s string) int {
	return strings.Index(s, `"query_vector":`)
}

func findClosingBrace(s string) string {
	depth := 0
	for i, ch := range s {
		if ch == '[' {
			depth++
		} else if ch == ']' {
			depth--
			if depth == 0 {
				return s[i:]
			}
		}
	}
	return ""
}

// ---------- MAIN ----------
// func main() {
// 	cfg := elasticsearch7.Config{}
// 	es, err := elasticsearch7.NewClient(cfg)

// 	server, err := initServer()
// 	if err != nil {
// 		log.Fatalf("❌ Failed to initialize server: %v", err)
// 	}
// 	defer server.Close()

// 	if err != nil {
// 		log.Fatalf("ES client error: %v", err)
// 	}

// 	if err := createIndex(es); err != nil {
// 		log.Printf("warning creating index: %v", err)
// 	}

// 	app := fiber.New()

// 	app.Post("/target_rules", func(c *fiber.Ctx) error {
// 		var rule TargetRule
// 		if err := c.BodyParser(&rule); err != nil {
// 			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
// 		}

// 		log.Println("========================================")
// 		log.Printf("Indexing Rule ID: %d", rule.ID)
// 		log.Printf("Target Criteria: %+v", rule.Target)

// 		keyfield, ok := rule.Target["keyword"]
// 		if !ok || len(keyfield.Values) == 0 {
// 			return c.Status(400).JSON(fiber.Map{"error": "keyword missing"})
// 		}

// 		indexed := 0
// 		for _, kw := range keyfield.Values {
// 			log.Printf("  - Processing keyword: %s", kw)
// 			vec, err := embedHandler(strings.TrimSpace(kw))
// 			if err != nil {
// 				log.Printf("    ERROR getting embedding: %v", err)
// 				return c.Status(500).JSON(fiber.Map{"error": err.Error()})
// 			}
// 			log.Printf("    Embedding vector length: %d", len(vec))

// 			if err := indexKeywordDoc(es, rule, kw, vec); err != nil {
// 				log.Printf("    ERROR indexing: %v", err)
// 				return c.Status(500).JSON(fiber.Map{"error": err.Error()})
// 			}
// 			log.Printf("    ✓ Indexed successfully")
// 			indexed++
// 		}

// 		log.Printf("Total indexed: %d keywords", indexed)
// 		log.Println("========================================")

// 		return c.JSON(fiber.Map{"message": "indexed", "id": rule.ID, "keywords": keyfield.Values, "count": indexed})
// 	})

// 	// POST /search
// 	app.Post("/search", func(c *fiber.Ctx) error {
// 		type SearchReq struct {
// 			Query     string                    `json:"query"`
// 			Variables map[string]TargetCriteria `json:"variables"`
// 			MinScore  float64                   `json:"min_score"` // Ngưỡng similarity tối thiểu
// 		}
// 		var req SearchReq
// 		if err := c.BodyParser(&req); err != nil {
// 			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
// 		}
// 		// Default min_score = 0.5 (lỏng hơn để test)
// 		// Sau khi test xong có thể tăng lên
// 		if req.MinScore == 0 {
// 			req.MinScore = 0.5 // Rất thấp để test, sau tăng lên 0.8-1.0
// 		}

// 		qvec, err := embedHandler(req.Query)
// 		if err != nil {
// 			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
// 		}

// 		esq := buildESQuery(qvec, req.Variables, req.MinScore, req.Query)
// 		body, _ := json.Marshal(esq)

// 		// Pretty print query để dễ đọc, nhưng truncate vector
// 		var prettyQuery bytes.Buffer
// 		json.Indent(&prettyQuery, body, "", "  ")

// 		// Truncate vector trong log để dễ đọc
// 		// prettyStr := prettyQuery.String()
// 		// vectorStart := ""
// 		// vectorEnd := ""
// 		// if len(qvec) > 0 {
// 		// 	vectorStart = fmt.Sprintf("[%.4f, %.4f, %.4f", qvec[0], qvec[1], qvec[2])
// 		// 	vectorEnd = fmt.Sprintf("%.4f, %.4f, %.4f]", qvec[len(qvec)-3], qvec[len(qvec)-2], qvec[len(qvec)-1])
// 		// }

// 		// log.Println("========================================")
// 		// log.Printf("Search Query: %s", req.Query)
// 		// log.Printf("Variables: %+v", req.Variables)
// 		// log.Printf("Min Score Threshold: %.2f", req.MinScore)
// 		// log.Printf("Query Vector: %s ... (384 dims) ... %s", vectorStart, vectorEnd)
// 		// log.Println("----------------------------------------")
// 		// log.Println("Elasticsearch Query (vector truncated for readability):")

// 		// // Replace long vector array with placeholder
// 		// shortQuery := prettyStr
// 		// if idx := findVectorInJSON(shortQuery); idx != -1 {
// 		// 	shortQuery = shortQuery[:idx] + `"query_vector": ` + vectorStart + ` ... 378 more values ... ` + vectorEnd + `
// 		//   }` + findClosingBrace(shortQuery[idx:])
// 		// }
// 		// log.Println(shortQuery)
// 		// log.Println("----------------------------------------")
// 		// log.Printf("To run in Kibana Dev Tools: GET /%s/_search", IndexName)
// 		// log.Println("(Use full vector from /debug/test-similarity for exact replication)")
// 		// log.Println("========================================")

// 		res, err := es.Search(
// 			es.Search.WithIndex(IndexName),
// 			es.Search.WithBody(bytes.NewReader(body)),
// 		)
// 		if err != nil {
// 			log.Printf("ES Search Error: %v", err)
// 			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
// 		}
// 		defer res.Body.Close()

// 		// Log response status
// 		log.Printf("ES Response Status: %s", res.Status())

// 		var out struct {
// 			Hits struct {
// 				Total struct {
// 					Value int `json:"value"`
// 				} `json:"total"`
// 				Hits []struct {
// 					Score  float64                `json:"_score"`
// 					Source map[string]interface{} `json:"_source"`
// 				} `json:"hits"`
// 			} `json:"hits"`
// 		}
// 		if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
// 			log.Printf("ES Response Decode Error: %v", err)
// 			return c.Status(500).JSON(fiber.Map{"error": err.Error()})
// 		}

// 		log.Printf("ES Total Hits: %d", out.Hits.Total.Value)

// 		type R struct {
// 			Keyword    string  `json:"keyword"`
// 			Score      float64 `json:"score"`
// 			Similarity float64 `json:"similarity"` // Thêm similarity % để dễ hiểu
// 			RuleID     int     `json:"rule_id"`
// 		}
// 		results := []R{}
// 		for _, h := range out.Hits.Hits {
// 			kw, _ := h.Source["original_keyword"].(string)
// 			rid := 0
// 			if v, ok := h.Source["rule_id"].(float64); ok {
// 				rid = int(v)
// 			}
// 			// Convert score về similarity percentage
// 			similarity := (h.Score - 1.0) * 100 // cosineSimilarity + 1.0, so -1.0 to get original
// 			results = append(results, R{
// 				Keyword:    kw,
// 				Score:      h.Score,
// 				Similarity: similarity,
// 				RuleID:     rid,
// 			})
// 			log.Printf("  - Keyword: %s, Score: %.4f, Similarity: %.2f%%, RuleID: %d", kw, h.Score, similarity, rid)
// 		}

// 		// Sort by score descending
// 		sort.Slice(results, func(i, j int) bool {
// 			return results[i].Score > results[j].Score
// 		})

// 		return c.JSON(fiber.Map{
// 			"total":     len(results),
// 			"results":   results,
// 			"min_score": req.MinScore,
// 			"query":     req.Query,
// 		})
// 	})

// 	port := os.Getenv("PORT")
// 	if port == "" {
// 		port = "9000"
// 	}
// 	log.Printf("Listening on :%s", port)
// 	log.Fatal(app.Listen(":" + port))
// }
