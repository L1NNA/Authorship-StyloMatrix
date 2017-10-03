/*******************************************************************************
 * Copyright 2015 McGill University. All rights reserved.                       
 *                                                                               
 * Unless required by applicable law or agreed to in writing, the software      
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF      
 * ANY KIND, either express or implied.                                         
 *******************************************************************************/
package ca.mcgill.sis.dmas.io;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.Charset;
import java.util.Iterator;
import java.util.zip.GZIPInputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Charsets;

import static ca.mcgill.sis.dmas.env.DmasApplication.applyDataContext;

public class LinesOnGzip extends Lines {

	private Logger logger = LoggerFactory.getLogger(LinesOnGzip.class);
	String file_;
	Charset charset;

	public LinesOnGzip(String file, Charset charset) throws IOException {
		file = applyDataContext(file);
		file_ = file;
		this.charset = charset;
	}

	@Override
	public Iterator<String> iterator() {
		return new LineIterator();
	}

	public class LineIterator implements Iterator<String> {

		private BufferedReader bReader = null;

		public LineIterator() {
			try {
				InputStream fileStream = new FileInputStream(file_);
				InputStream gzipStream = new GZIPInputStream(fileStream);
				Reader decoder = new InputStreamReader(gzipStream, charset);
				bReader = new BufferedReader(decoder);
				
			} catch (Exception e) {
				logger.error("Failed to open file.", e);
			}
		}

		String line = null;

		volatile boolean closed = false;

		@Override
		public boolean hasNext() {
			if (closed)
				return false;
			if (line != null)
				return true;

			try {
				line = bReader.readLine();
				if (line == null)
					return false;
				else {
					return true;
				}
			} catch (IOException e) {
				logger.error("Unable to read from file", e);
				try {
					closed = true;
					bReader.close();
				} catch (Exception e1) {
				}
				return false;
			}
		}

		@Override
		public String next() {
			String nextLine = line;
			line = null;
			return nextLine;
		}

		@Override
		public void remove() {
			logger.error("Unable to remove element. This is an immutable iterator.");
		}

	}

}
