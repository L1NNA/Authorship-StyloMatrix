/*******************************************************************************
 * Copyright 2015 McGill University. All rights reserved.                       
 *                                                                               
 * Unless required by applicable law or agreed to in writing, the software      
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF      
 * ANY KIND, either express or implied.                                         
 *******************************************************************************/
package ca.mcgill.sis.dmas.io;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableList;
import com.google.common.io.Files;

import static ca.mcgill.sis.dmas.env.DmasApplication.applyDataContext;

public class LinesOnRAM extends Lines {

	private Logger logger = LoggerFactory.getLogger(LinesOnRAM.class);

	String[] LINES = null;
	
	public LinesOnRAM(String file) throws IOException {
		file = applyDataContext(file);
		ImmutableList<String> list = Files.asCharSource(new File(file),
				charset).readLines();
		LINES = list.toArray(new String[list.size()]);
	}

	public LinesOnRAM(String file, Charset charset) throws IOException {
		file = applyDataContext(file);
		ImmutableList<String> list = Files.asCharSource(new File(file),
				charset).readLines();
		this.charset = charset;
		LINES = list.toArray(new String[list.size()]);
	}
	
	Charset charset = Charset.defaultCharset();

	@Override
	public Iterator<String> iterator() {
		return new LineIterator();
	}

	public class LineIterator implements Iterator<String> {

		public LineIterator() {
			index = 0;
		}

		int index = 0;

		@Override
		public boolean hasNext() {
			if (index < LINES.length && index >= 0) {
				return true;
			} else {
				return false;
			}
		}

		String line = null;
		@Override
		public String next() {
			line = LINES[index];
			index++;
			return line;
		}

		@Override
		public void remove() {
			logger.error("Unable to remove element. This is an immutable iterator.");
		}

	}

}
